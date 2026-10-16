# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint limits, velocity limits and joint drives of SolverSteel."""

import inspect
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    allocate_joint_velocity_limit_slots,
    allocate_rigid_velocity_limit_slots,
    build_joint_limit_rows,
    prescale_joint_velocity_limits,
)
from newton._src.solvers.steel.solver_steel import _get_joint_limit_warp_kernel, _use_resident_mfgs_metadata
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _built_rows(device, q: float, *, gap: float, lower: float = -1.0, upper: float = 1.0, scalar_builder=False):
    """Build the joint-limit rows of one single-DOF articulation and return ``(J, phi)``.

    ``scalar_builder`` selects the one-thread-per-articulation Warp builder of the CPU path
    instead of the one-warp-per-articulation CUDA builder.
    """
    max_constraints = 8
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    J_group = wp.zeros((1, max_constraints, 1), dtype=wp.float32, device=device)
    world_phi = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    index = wp.array([0], dtype=wp.int32, device=device)
    inputs = [
        index,
        index,
        index,
        index,
        wp.array([lower], dtype=wp.float32, device=device),
        wp.array([upper], dtype=wp.float32, device=device),
        wp.array([q], dtype=wp.float32, device=device),
        gap,
        max_constraints,
    ]
    outputs = [
        world_slot_counter,
        J_group,
        wp.zeros((1, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((1, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
        world_phi,
        wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
    ]
    if scalar_builder:
        wp.launch(build_joint_limit_rows, dim=1, inputs=[*inputs, 1], outputs=outputs, device=device)
    else:
        kernel = _get_joint_limit_warp_kernel(1, wp.get_device(device).arch, 1)
        wp.launch_tiled(kernel, dim=[1], inputs=[1, *inputs], outputs=outputs, block_dim=32, device=device)
    count = int(world_slot_counter.numpy()[0])
    return J_group.numpy()[0, :count, 0].tolist(), world_phi.numpy()[0, :count].tolist()


def _make_layout_scene(
    device, *, enable_joint_velocity_limits=True, solve_mode="matrix_free", response="immediate", iterations=0
):
    """Build a scene that produces every row family: limits, velocity limits and contacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverSteel.register_custom_attributes(builder)
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
            "steel:rigid_body_max_linear_velocity": 1.0,
            "steel:rigid_body_max_angular_velocity": 1.0,
        },
    )
    builder.add_shape_box(free_body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    velocity_limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
    velocity_limits[0] = 0.25
    model.joint_velocity_limit.assign(velocity_limits)
    solver = SolverSteel(
        model,
        solve_mode=solve_mode,
        enable_joint_limits=True,
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        velocity_limit_activation_fraction=0.5,
        articulation_constraint_max=64,
        free_body_constraint_max=64,
        iterations=iterations,
        articulated_contact_response=response,
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
    return model, solver, state_0, state_1, pipeline


def _make_layout_run(device, *, enable_joint_velocity_limits=True, solve_mode="matrix_free"):
    """Build the layout scene and its rows for one step."""
    model, solver, state_0, state_1, pipeline = _make_layout_scene(
        device, enable_joint_velocity_limits=enable_joint_velocity_limits, solve_mode=solve_mode
    )
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 60.0)
    return solver


def test_combined_row_families_follow_the_documented_layout(test, device):
    """Lay out dense rows as limits, velocity limits, contacts; free-body rows as contacts, velocity limits."""
    solver = _make_layout_run(device)
    dense_count = int(solver._constraint_count.numpy()[0])
    dense_types = solver._row_type.numpy()[0, :dense_count]
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

    mf_count = int(solver._mf_constraint_count.numpy()[0])
    mf_types = solver._mf_row_type.numpy()[0, :mf_count]
    mf_contact_end = int(solver._mf_contact_rows_end.numpy()[0])
    mf_contacts = np.flatnonzero((mf_types == PGS_CONSTRAINT_TYPE_CONTACT) | (mf_types == PGS_CONSTRAINT_TYPE_FRICTION))
    mf_velocity_limits = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertGreater(mf_contacts.size, 0)
    test.assertGreater(mf_velocity_limits.size, 0)
    test.assertTrue(np.all(mf_contacts < mf_contact_end))
    test.assertTrue(np.all(mf_velocity_limits >= mf_contact_end))


def test_dense_row_families_respect_phase_bounds(test, device):
    """Lay out split-mode dense rows as limits, contacts; free-body rows as contacts, velocity limits."""
    solver = _make_layout_run(device, enable_joint_velocity_limits=False, solve_mode="split")
    dense_count = int(solver._constraint_count.numpy()[0])
    dense_types = solver._row_type.numpy()[0, :dense_count]
    limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    contact_rows = np.flatnonzero(
        (dense_types == PGS_CONSTRAINT_TYPE_CONTACT) | (dense_types == PGS_CONSTRAINT_TYPE_FRICTION)
    )
    test.assertGreater(limit_rows.size, 0)
    test.assertGreater(contact_rows.size, 0)
    test.assertEqual(int(np.sum(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)), 0)
    test.assertLess(limit_rows.max(), contact_rows.min())

    mf_count = int(solver._mf_constraint_count.numpy()[0])
    mf_types = solver._mf_row_type.numpy()[0, :mf_count]
    mf_contact_end = int(solver._mf_contact_rows_end.numpy()[0])
    mf_contacts = np.flatnonzero((mf_types == PGS_CONSTRAINT_TYPE_CONTACT) | (mf_types == PGS_CONSTRAINT_TYPE_FRICTION))
    mf_velocity_limits = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertGreater(mf_contacts.size, 0)
    test.assertGreater(mf_velocity_limits.size, 0)
    test.assertTrue(np.all(mf_contacts < mf_contact_end))
    test.assertTrue(np.all(mf_velocity_limits >= mf_contact_end))


def test_velocity_limit_rows_require_the_option(test, device):
    """Build no joint velocity-limit rows unless velocity limits are enabled."""
    solver = _make_layout_run(device, enable_joint_velocity_limits=False)
    dense_count = int(solver._constraint_count.numpy()[0])
    dense_types = solver._row_type.numpy()[0, :dense_count]
    test.assertEqual(int(np.sum(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)), 0)
    test.assertGreater(int(np.sum(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)), 0)


def test_finite_gap_builds_only_near_limit_rows(test, device, scalar_builder=False):
    """Create a limit row only within the activation gap of that limit."""
    test.assertEqual(_built_rows(device, 0.0, gap=0.2, scalar_builder=scalar_builder), ([], []))
    jacobian, phi = _built_rows(device, -0.85, gap=0.2, scalar_builder=scalar_builder)
    test.assertEqual(jacobian, [1.0])
    test.assertAlmostEqual(phi[0], 0.15, places=6)
    jacobian, phi = _built_rows(device, 0.85, gap=0.2, scalar_builder=scalar_builder)
    test.assertEqual(jacobian, [-1.0])
    test.assertAlmostEqual(phi[0], 0.15, places=6)


def test_finite_gap_does_not_activate_unlimited_sentinel_limits(test, device, scalar_builder=False):
    """Treat the builder's unlimited sentinel as no limit."""
    test.assertEqual(
        _built_rows(device, 0.0, gap=0.2, lower=-MAXVAL, upper=MAXVAL, scalar_builder=scalar_builder), ([], [])
    )


def test_infinite_gap_allocates_every_finite_limit(test, device, scalar_builder=False):
    """Create both rows of every finite limit with the default infinite gap."""
    jacobian, phi = _built_rows(device, 0.0, gap=float("inf"), scalar_builder=scalar_builder)
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
    solver = SolverSteel(model, iterations=8, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    control.joint_target_q.fill_(1.0)
    for _ in range(240):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return float(state_0.joint_q.numpy()[0])


def test_limit_holds_a_driven_joint(test, device, solve_mode="matrix_free"):
    """Hold a joint driven past its upper limit at the limit when joint limits are enabled."""
    position = _driven_limited_joint_position(device, solve_mode=solve_mode, enable_joint_limits=True)
    test.assertAlmostEqual(position, 0.3, delta=2.0e-3)


def test_disabled_joint_limits_are_not_enforced(test, device, solve_mode="matrix_free"):
    """Leave finite joint limits unenforced by default, so the drive reaches its target."""
    test.assertAlmostEqual(_driven_limited_joint_position(device, solve_mode=solve_mode), 1.0, delta=2.0e-3)
    position = _driven_limited_joint_position(device, solve_mode=solve_mode, enable_joint_limits=False)
    test.assertAlmostEqual(position, 1.0, delta=2.0e-3)


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
                solver = SolverSteel(model, enable_joint_limits=True, warn_constraint_overflow=False)
            messages = [str(w.message) for w in caught if "joint position limits" in str(w.message)]
            test.assertEqual(len(messages), int(expect_warning), messages)
            if expect_warning:
                test.assertIn("at least 34 dense rows", messages[0])
                test.assertIn("articulation_constraint_max=32", messages[0])
            solver.step(model.state(), model.state(), model.control(), None, 0.01)
            # Large finite bounds are not reclassified as unlimited: two rows per DOF, up to
            # the capacity.
            test.assertEqual(int(solver._constraint_count.numpy()[0]), min(2 * count, 32))
            test.assertEqual(bool(solver.constraint_overflow.numpy()[0]), expect_warning)

    model = _build_default_limit_chain(device, 17)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SolverSteel(model, enable_joint_limits=True, articulation_constraint_max=34)
        gap_solver = SolverSteel(model, enable_joint_limits=True, joint_limit_activation_gap=0.5)
    test.assertFalse([w for w in caught if "joint position limits" in str(w.message)])
    gap_solver.step(model.state(), model.state(), model.control(), None, 0.01)
    test.assertEqual(int(gap_solver._constraint_count.numpy()[0]), 0)
    test.assertFalse(bool(gap_solver.constraint_overflow.numpy()[0]))


def test_disabled_joint_limits_build_no_rows(test, device, **mode_kwargs):
    """Build no limit rows and use no capacity for finite limits when joint limits are disabled."""
    model = _build_default_limit_chain(device, 17)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solvers = (
            SolverSteel(model, **mode_kwargs),
            SolverSteel(model, enable_joint_limits=False, **mode_kwargs),
        )
    # The 34 rows of the +/-1e10 defaults would exceed the default capacity of 32 if limits were enabled.
    test.assertFalse([w for w in caught if "joint position limits" in str(w.message)])
    for solver in solvers:
        test.assertFalse(solver.enable_joint_limits)
        solver.step(model.state(), model.state(), model.control(), None, 0.01)
        test.assertEqual(int(solver._constraint_count.numpy()[0]), 0)
        test.assertFalse(bool(solver.constraint_overflow.numpy()[0]))

    # A joint outside its limit gets no row either, whatever the activation gap.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    joint = builder.add_joint_revolute(-1, link, axis=newton.Axis.Z, limit_lower=-0.1, limit_upper=0.1)
    builder.add_articulation([joint])
    builder.joint_q[0] = 0.2
    model = builder.finalize(device=device)
    solver = SolverSteel(model, joint_limit_activation_gap=0.5, **mode_kwargs)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, 0.01)
    test.assertEqual(int(solver._constraint_count.numpy()[0]), 0)
    test.assertAlmostEqual(float(state_1.joint_q.numpy()[0]), 0.2, places=6)


def _run_layout_trajectory(device, response, iterations):
    """Step the layout scene four times; record each step's row families, impulses and joint state."""
    model, solver, state_0, state_1, pipeline = _make_layout_scene(device, response=response, iterations=iterations)
    contacts = pipeline.contacts()
    control = model.control()
    samples = []
    for _ in range(4):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        dense_count = int(solver._constraint_count.numpy()[0])
        mf_count = int(solver._mf_constraint_count.numpy()[0])
        samples.append(
            {
                "dense_count": dense_count,
                "row_type": solver._row_type.numpy()[0, :dense_count].copy(),
                "impulses": solver._impulses.numpy()[0, :dense_count].copy(),
                "mf_row_type": solver._mf_row_type.numpy()[0, :mf_count].copy(),
                "propagation_count": int(solver._propagation_constraint_count.numpy()[0]),
                "joint_q": state_1.joint_q.numpy().copy(),
                "joint_qd": state_1.joint_qd.numpy().copy(),
            }
        )
        state_0, state_1 = state_1, state_0
    return samples


def test_fused_propagation_matches_phased_trajectory(test, device):
    """Solve every row family in the propagation schedule; the fused kernel matches the phased launches.

    Each propagation iteration solves the dense joint-limit rows, the contact rows, the
    propagation rows and their tree propagation, then the velocity-limit rows. The scene's
    articulated link (on its limit, against the ground) and free body (with velocity limits)
    do not interact, so the separately launched phases and the fused kernel, which keeps the
    free body's contacts on free-body rows, must give the same trajectory. A single iteration
    pins the order of the row families within an iteration.
    """
    for iterations in (1, 8):
        with test.subTest(iterations=iterations):
            _check_fused_matches_phased(test, device, iterations)


def _check_fused_matches_phased(test, device, iterations):
    phased = _run_layout_trajectory(device, "propagation", iterations)
    fused = _run_layout_trajectory(device, "propagation_fused", iterations)
    for step, (expected, observed) in enumerate(zip(phased, fused, strict=True)):
        for label, sample in (("propagation", expected), ("propagation_fused", observed)):
            with test.subTest(step=step, response=label):
                types = sample["row_type"]
                # Dense rows hold only the joint rows; every contact touching the link is a propagation row.
                test.assertGreater(int(np.sum(types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)), 0)
                test.assertTrue(
                    np.all(
                        (types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT) | (types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
                    )
                )
                test.assertGreater(sample["propagation_count"], 0)
                test.assertGreater(int(np.sum(sample["mf_row_type"] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)), 0)
        np.testing.assert_array_equal(observed["row_type"], expected["row_type"], err_msg=f"step {step}")
        # The free body's contacts are propagation rows in the phased response and free-body rows when fused.
        test.assertEqual(int(np.sum(expected["mf_row_type"] == PGS_CONSTRAINT_TYPE_CONTACT)), 0)
        test.assertGreater(int(np.sum(observed["mf_row_type"] == PGS_CONSTRAINT_TYPE_CONTACT)), 0)
        # The fused kernel solves the free body's contacts as free-body rows, which rounds differently.
        for label in ("impulses", "joint_q", "joint_qd"):
            np.testing.assert_allclose(
                observed[label], expected[label], rtol=1.0e-5, atol=1.0e-5, err_msg=f"{label} differed at step {step}"
            )
    limit_rows = np.flatnonzero(phased[0]["row_type"] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    test.assertGreater(float(np.max(np.abs(phased[0]["impulses"][limit_rows]))), 0.0)


class TestSteelJointLimitActivationGap(unittest.TestCase):
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
    ("test_fused_propagation_matches_phased_trajectory", test_fused_propagation_matches_phased_trajectory),
):
    add_function_test(TestSteelJointLimitActivationGap, _name, _func, devices=devices)


split_devices = get_test_devices()


add_function_test(
    TestSteelJointLimitActivationGap,
    "test_dense_row_families_respect_phase_bounds",
    test_dense_row_families_respect_phase_bounds,
    devices=split_devices,
)


for _name, _func in (
    ("test_limit_holds_a_driven_joint", test_limit_holds_a_driven_joint),
    ("test_disabled_joint_limits_are_not_enforced", test_disabled_joint_limits_are_not_enforced),
    ("test_disabled_joint_limits_build_no_rows", test_disabled_joint_limits_build_no_rows),
):
    add_function_test(
        TestSteelJointLimitActivationGap,
        f"{_name}_split",
        _func,
        devices=split_devices,
        solve_mode="split",
    )


for _name, _func in (
    ("test_finite_gap_builds_only_near_limit_rows", test_finite_gap_builds_only_near_limit_rows),
    (
        "test_finite_gap_does_not_activate_unlimited_sentinel_limits",
        test_finite_gap_does_not_activate_unlimited_sentinel_limits,
    ),
    ("test_infinite_gap_allocates_every_finite_limit", test_infinite_gap_allocates_every_finite_limit),
):
    add_function_test(
        TestSteelJointLimitActivationGap,
        f"{_name}_scalar_builder",
        _func,
        devices=split_devices,
        scalar_builder=True,
    )


def test_warp_builder_matches_reference_row_order_and_values(test, device, scalar_builder=False):
    """Emit active limit rows in DOF order with the lower row before the upper row of each DOF.

    ``scalar_builder`` checks the one-thread-per-articulation builder of the CPU path instead.
    """
    articulation_count = 257
    size = 6
    max_constraints = 2 * size
    dof_count = articulation_count * size
    gap = 0.25
    rng = np.random.default_rng(42)
    q = rng.uniform(-1.2, 1.2, dof_count).astype(np.float32)
    lower = np.full(dof_count, -1.0, dtype=np.float32)
    upper = np.full(dof_count, 1.0, dtype=np.float32)

    outputs = (
        wp.zeros(articulation_count, dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints, size), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
    )
    inputs = [
        wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device),
        wp.array(np.arange(articulation_count), dtype=wp.int32, device=device),
        wp.array(np.arange(articulation_count), dtype=wp.int32, device=device),
        wp.array(np.arange(dof_count), dtype=wp.int32, device=device),
        wp.array(lower, dtype=wp.float32, device=device),
        wp.array(upper, dtype=wp.float32, device=device),
        wp.array(q, dtype=wp.float32, device=device),
        gap,
        max_constraints,
    ]
    if scalar_builder:
        wp.launch(
            build_joint_limit_rows,
            dim=articulation_count,
            inputs=[*inputs, size],
            outputs=list(outputs),
            device=device,
        )
    else:
        warps_per_block = 4
        kernel = _get_joint_limit_warp_kernel(size, wp.get_device(device).arch, warps_per_block)
        wp.launch_tiled(
            kernel,
            dim=[(articulation_count + warps_per_block - 1) // warps_per_block],
            inputs=[articulation_count, *inputs],
            outputs=list(outputs),
            block_dim=32 * warps_per_block,
            device=device,
        )
    counter, J_group, row_type, row_parent, row_mu, phi, target = (array.numpy() for array in outputs)

    for art in range(articulation_count):
        expected_J = np.zeros((max_constraints, size), dtype=np.float32)
        expected_phi = []
        for local in range(size):
            dof = art * size + local
            for sign, value, active in (
                (1.0, q[dof] - lower[dof], q[dof] <= lower[dof] + gap),
                (-1.0, upper[dof] - q[dof], q[dof] >= upper[dof] - gap),
            ):
                if active:
                    expected_J[len(expected_phi), local] = sign
                    expected_phi.append(value)
        count = len(expected_phi)
        test.assertEqual(int(counter[art]), count)
        np.testing.assert_array_equal(J_group[art], expected_J)
        np.testing.assert_array_equal(row_type[art, :count], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
        np.testing.assert_array_equal(row_parent[art, :count], -1)
        np.testing.assert_array_equal(row_mu[art, :count], 0.0)
        np.testing.assert_array_equal(target[art, :count], 0.0)
        np.testing.assert_allclose(phi[art, :count], np.asarray(expected_phi, dtype=np.float32), rtol=0.0, atol=1.0e-7)


class TestSteelJointLimitWarp(unittest.TestCase):
    pass


add_function_test(
    TestSteelJointLimitWarp,
    "test_warp_builder_matches_reference_row_order_and_values",
    test_warp_builder_matches_reference_row_order_and_values,
    devices=get_cuda_test_devices(),
)


add_function_test(
    TestSteelJointLimitWarp,
    "test_scalar_builder_matches_reference_row_order_and_values",
    test_warp_builder_matches_reference_row_order_and_values,
    devices=get_test_devices(),
    scalar_builder=True,
)


QDOT_MAX = 0.5


# Slack on the limit for an under-converged sweep; the final pass leaves the DOF at the limit.
LIMIT_TOL = 1.05


def _allocated_joint_velocity_slots(device, qd: float, *, fraction: float, qdot_max: float = 1.0):
    velocity_limit_slot = wp.full((2,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((2,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array([qdot_max], dtype=wp.float32, device=device),
            wp.array([qd], dtype=wp.float32, device=device),
            fraction,
            wp.array([-1], dtype=wp.int32, device=device),
            0,
            wp.array([0], dtype=wp.int32, device=device),
            8,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )
    return (
        velocity_limit_slot.numpy().tolist(),
        velocity_limit_sign.numpy().tolist(),
        int(world_slot_counter.numpy()[0]),
    )


def _allocated_rigid_velocity_slots(device, qd6, *, fraction: float, lin_limit: float = 1.0, ang_limit: float = 1.0):
    rigid_velocity_limit_slot = wp.full((12,), -1, dtype=wp.int32, device=device)
    rigid_velocity_limit_sign = wp.zeros((12,), dtype=wp.float32, device=device)
    mf_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_rigid_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(BodyFlags.DYNAMIC)], dtype=wp.int32, device=device),
            wp.array([lin_limit], dtype=wp.float32, device=device),
            wp.array([ang_limit], dtype=wp.float32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array(list(qd6), dtype=wp.float32, device=device),
            fraction,
            64,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[rigid_velocity_limit_slot, rigid_velocity_limit_sign, mf_slot_counter],
        device=device,
    )
    return (
        rigid_velocity_limit_slot.numpy().tolist(),
        rigid_velocity_limit_sign.numpy().tolist(),
        int(mf_slot_counter.numpy()[0]),
    )


def _build_arm_and_box_model(device) -> newton.Model:
    """A driven revolute arm plus a heavy free box overlapping its tip, without gravity or friction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    # Arm spanning x in [0, 0.8] at z = 0.5, hinged at its left end about y.
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    j_arm = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_arm])
    # Heavy box, bottom face slightly penetrating the arm top near the tip.
    box = builder.add_link(xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
    return builder.finalize(device=device)


def _max_arm_speed_over_impact(device, *, enable_joint_velocity_limits: bool) -> float:
    model = _build_arm_and_box_model(device)
    n = model.joint_dof_count
    target_ke = np.zeros(n, dtype=np.float32)
    target_kd = np.zeros(n, dtype=np.float32)
    vel_limit = np.full(n, np.inf, dtype=np.float32)
    target_ke[0] = 200.0
    target_kd[0] = 5.0
    vel_limit[0] = QDOT_MAX
    model.joint_target_ke.assign(target_ke)
    model.joint_target_kd.assign(target_kd)
    model.joint_velocity_limit.assign(vel_limit)
    # Deliberately under-converged: the velocity limit must still have the last word.
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        iterations=8,
        articulation_constraint_max=32,
        free_body_constraint_max=32,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[3] = -3.0  # box linear z: slam into the arm tip
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    max_arm_speed = 0.0
    for _ in range(20):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        max_arm_speed = max(max_arm_speed, float(abs(state_0.joint_qd.numpy()[0])))
    return max_arm_speed


def test_dedicated_rows_hold_limit_through_impact(test, device):
    """Hold a driven joint's velocity limit through an impact that would exceed it several times over."""
    unlimited = _max_arm_speed_over_impact(device, enable_joint_velocity_limits=False)
    limited = _max_arm_speed_over_impact(device, enable_joint_velocity_limits=True)
    test.assertGreater(unlimited, 2.0 * QDOT_MAX, "the impact must exceed the limit when it is not enforced")
    test.assertLessEqual(limited, QDOT_MAX * LIMIT_TOL)


def test_fraction_zero_allocates_every_joint_row(test, device):
    """Allocate both rows of every limited DOF regardless of its velocity when the fraction is zero."""
    for qd in (0.0, 0.5, -2.0):
        with test.subTest(qd=qd):
            slots, signs, count = _allocated_joint_velocity_slots(device, qd, fraction=0.0)
            test.assertEqual(slots, [0, 1])
            test.assertEqual(signs, [1.0, -1.0])
            test.assertEqual(count, 2)


def test_fraction_gates_static_joint_dof_to_zero_rows(test, device):
    """Allocate no rows for a DOF below the activation fraction of its limit."""
    slots, signs, count = _allocated_joint_velocity_slots(device, 0.0, fraction=0.9)
    test.assertEqual(slots, [-1, -1])
    test.assertEqual(signs, [0.0, 0.0])
    test.assertEqual(count, 0)
    slots, signs, count = _allocated_joint_velocity_slots(device, 0.85, fraction=0.9)
    test.assertEqual(slots, [-1, -1])
    test.assertEqual(count, 0)


def test_joint_dof_past_threshold_allocates_rows_same_step(test, device):
    """Allocate the row pair in the same step a DOF crosses the activation fraction."""
    for qd in (0.95, -0.95, 1.5):
        with test.subTest(qd=qd):
            slots, signs, count = _allocated_joint_velocity_slots(device, qd, fraction=0.9)
            test.assertEqual(slots, [0, 1])
            test.assertEqual(signs, [1.0, -1.0])
            test.assertEqual(count, 2)


def test_fraction_zero_allocates_every_rigid_row(test, device):
    """Allocate all twelve free-body velocity-limit rows when the fraction is zero."""
    slots, signs, count = _allocated_rigid_velocity_slots(device, [0.0] * 6, fraction=0.0)
    test.assertEqual(slots, list(range(12)))
    test.assertEqual(signs, [1.0, -1.0] * 6)
    test.assertEqual(count, 12)


def test_fraction_gates_static_rigid_body_to_zero_rows(test, device):
    """Allocate no free-body rows for a body below the activation fraction."""
    slots, signs, count = _allocated_rigid_velocity_slots(device, [0.0] * 6, fraction=0.9)
    test.assertEqual(slots, [-1] * 12)
    test.assertEqual(signs, [0.0] * 12)
    test.assertEqual(count, 0)


def test_rigid_axis_past_threshold_allocates_rows_same_step(test, device):
    """Allocate only the row pair of the free-body axis past the activation fraction."""
    slots, signs, count = _allocated_rigid_velocity_slots(device, [0.0, 0.0, 0.95, 0.0, 0.0, 0.0], fraction=0.9)
    expected_slots = [-1] * 12
    expected_slots[4] = 0
    expected_slots[5] = 1
    expected_signs = [0.0] * 12
    expected_signs[4] = 1.0
    expected_signs[5] = -1.0
    test.assertEqual(slots, expected_slots)
    test.assertEqual(signs, expected_signs)
    test.assertEqual(count, 2)


def test_free_body_velocity_limits_hold(test, device, solve_mode="matrix_free"):
    """Hold a free body's authored linear and angular speed limits."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverSteel.register_custom_attributes(builder)
    body = builder.add_body(
        custom_attributes={
            "steel:rigid_body_max_linear_velocity": 0.5,
            "steel:rigid_body_max_angular_velocity": 1.0,
        }
    )
    builder.add_shape_box(body, hx=0.1, hy=0.2, hz=0.3)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    state_0, state_1 = model.state(), model.state()
    qd = state_0.joint_qd.numpy()
    qd[3:6] = (5.0, 0.0, 0.0)
    state_0.joint_qd.assign(qd)
    for _ in range(60):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    final = state_0.joint_qd.numpy()
    test.assertLessEqual(float(np.max(np.abs(final[0:3]))), 0.5 * LIMIT_TOL)
    test.assertLessEqual(float(np.max(np.abs(final[3:6]))), 1.0 * LIMIT_TOL)


class TestSteelVelocityLimitActivationFraction(unittest.TestCase):
    pass


for _name, _func in (
    ("test_dedicated_rows_hold_limit_through_impact", test_dedicated_rows_hold_limit_through_impact),
    ("test_fraction_zero_allocates_every_joint_row", test_fraction_zero_allocates_every_joint_row),
    ("test_fraction_gates_static_joint_dof_to_zero_rows", test_fraction_gates_static_joint_dof_to_zero_rows),
    ("test_joint_dof_past_threshold_allocates_rows_same_step", test_joint_dof_past_threshold_allocates_rows_same_step),
    ("test_fraction_zero_allocates_every_rigid_row", test_fraction_zero_allocates_every_rigid_row),
    ("test_fraction_gates_static_rigid_body_to_zero_rows", test_fraction_gates_static_rigid_body_to_zero_rows),
    (
        "test_rigid_axis_past_threshold_allocates_rows_same_step",
        test_rigid_axis_past_threshold_allocates_rows_same_step,
    ),
    ("test_free_body_velocity_limits_hold", test_free_body_velocity_limits_hold),
):
    add_function_test(TestSteelVelocityLimitActivationFraction, _name, _func, devices=devices)


# The allocators are mode-independent kernels; free-body velocity limits are also rows in split mode.
for _name, _func in (
    ("test_fraction_zero_allocates_every_joint_row", test_fraction_zero_allocates_every_joint_row),
    ("test_fraction_gates_static_joint_dof_to_zero_rows", test_fraction_gates_static_joint_dof_to_zero_rows),
    ("test_joint_dof_past_threshold_allocates_rows_same_step", test_joint_dof_past_threshold_allocates_rows_same_step),
    ("test_fraction_zero_allocates_every_rigid_row", test_fraction_zero_allocates_every_rigid_row),
    ("test_fraction_gates_static_rigid_body_to_zero_rows", test_fraction_gates_static_rigid_body_to_zero_rows),
    (
        "test_rigid_axis_past_threshold_allocates_rows_same_step",
        test_rigid_axis_past_threshold_allocates_rows_same_step,
    ),
):
    add_function_test(
        TestSteelVelocityLimitActivationFraction, _name, _func, devices=[d for d in get_test_devices() if d.is_cpu]
    )


add_function_test(
    TestSteelVelocityLimitActivationFraction,
    "test_free_body_velocity_limits_hold_split",
    test_free_body_velocity_limits_hold,
    devices=get_test_devices(),
    solve_mode="split",
)


def _max_arm_speed_and_fused_over_impact(
    device, *, enable_joint_velocity_limits: bool, fuse: bool
) -> tuple[float, int]:
    model = _build_arm_and_box_model(device)
    n = model.joint_dof_count
    target_ke = np.zeros(n, dtype=np.float32)
    target_kd = np.zeros(n, dtype=np.float32)
    vel_limit = np.full(n, np.inf, dtype=np.float32)
    target_ke[0] = 200.0
    target_kd[0] = 5.0
    vel_limit[0] = QDOT_MAX
    model.joint_target_ke.assign(target_ke)
    model.joint_target_kd.assign(target_kd)
    model.joint_velocity_limit.assign(vel_limit)
    # Deliberately under-converged: ordering only matters when the sweep does not converge.
    # A clamp inside the drive-row visit lets this impact reach about 1.5x the limit at 8
    # iterations; dedicated rows and the end-of-iteration clamp hold it.
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        drive_mode="rows",
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        fuse_joint_velocity_limits=fuse,
        iterations=8,
        articulation_constraint_max=32,
        free_body_constraint_max=32,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[3] = -3.0  # box linear z: slam into the arm tip
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    max_arm_speed = 0.0
    for _ in range(20):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        max_arm_speed = max(max_arm_speed, float(abs(state_0.joint_qd.numpy()[0])))
    return max_arm_speed, int(solver.velocity_limits_fused)


def test_dedicated_rows_hold_pgs_drive_limit_through_impact(test, device):
    """Hold a PGS-driven joint's velocity limit through the impact with dedicated velocity-limit rows."""
    unlimited, _ = _max_arm_speed_and_fused_over_impact(device, enable_joint_velocity_limits=False, fuse=False)
    limited, fused = _max_arm_speed_and_fused_over_impact(device, enable_joint_velocity_limits=True, fuse=False)
    test.assertFalse(fused)
    test.assertGreater(unlimited, 2.0 * QDOT_MAX, "the impact must exceed the limit when it is not enforced")
    test.assertLessEqual(limited, QDOT_MAX * LIMIT_TOL)


def test_fused_clamp_holds_limit_through_impact(test, device):
    """Hold the same limit with the fused end-of-iteration clamp that replaces the rows."""
    speed, fused = _max_arm_speed_and_fused_over_impact(device, enable_joint_velocity_limits=True, fuse=True)
    test.assertTrue(fused)
    test.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)


class TestSteelFusedVelocityLimitOrdering(unittest.TestCase):
    pass


for _name, _func in (
    ("test_dedicated_rows_hold_limit_through_impact", test_dedicated_rows_hold_pgs_drive_limit_through_impact),
    ("test_fused_clamp_holds_limit_through_impact", test_fused_clamp_holds_limit_through_impact),
):
    add_function_test(TestSteelFusedVelocityLimitOrdering, _name, _func, devices=devices)


DT = 1.0 / 120.0


MASS = 2.0


ARMATURE = 0.1


KE = 400.0


KD = 15.0


TARGET_POS = 0.3


TARGET_VEL = 0.2


def _build_slider(device, *, effort_limit=1.0e6):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=MASS, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=newton.Axis.X,
        target_ke=KE,
        target_kd=KD,
        target_pos=TARGET_POS,
        target_vel=TARGET_VEL,
        armature=ARMATURE,
        effort_limit=effort_limit,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _backward_euler_trajectory(steps):
    """Implicit PD drive: ``(m + a + dt kd + dt^2 ke) qdd = -(ke (q - q_t + dt qd) + kd (qd - qd_t))``."""
    q = 0.0
    qd = 0.0
    history = []
    effective_mass = MASS + ARMATURE + DT * KD + DT * DT * KE
    for _ in range(steps):
        u0 = -(KE * (q - TARGET_POS + DT * qd) + KD * (qd - TARGET_VEL))
        qd = qd + DT * u0 / effective_mass
        q = q + DT * qd
        history.append((q, qd))
    return np.asarray(history)


def _slider_trajectory(model, steps, **kwargs):
    solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", **kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    history = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        history.append((float(state_0.joint_q.numpy()[0]), float(state_0.joint_qd.numpy()[0])))
    return np.asarray(history)


def _build_driven_chain(device, num_links=3, num_worlds=2):
    chain = newton.ModelBuilder()
    joints = []
    parent = -1
    for i in range(num_links):
        link = chain.add_link()
        chain.add_shape_box(link, hx=0.2, hy=0.05, hz=0.05)
        joints.append(
            chain.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0 if i == 0 else 0.25, 0.0, 2.0 if i == 0 else 0.0)),
                child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0)),
                target_ke=100.0,
                target_kd=5.0,
                limit_lower=-1.0,
                limit_upper=1.0,
            )
        )
        parent = link
    chain.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(chain, num_worlds, spacing=(2.0, 0.0, 0.0))
    return builder.finalize(device=device)


def _build_arm_and_box_model_pgs_drives(device) -> newton.Model:
    """A driven, velocity-limited revolute arm and a free box hitting its tip, without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    j_arm = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        target_ke=200.0,
        target_kd=5.0,
    )
    builder.add_articulation([j_arm])
    box = builder.add_link(xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
    model = builder.finalize(device=device)
    limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
    limits[0] = 0.5
    model.joint_velocity_limit.assign(limits)
    return model


def _run_arm_and_box(model, solver, steps, *, capture=False):
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[3] = -3.0
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    def substeps():
        # An even number of substeps keeps the state roles fixed for replay.
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        pipeline.collide(state_1, contacts)
        solver.step(state_1, state_0, control, contacts, 1.0 / 240.0)

    if capture:
        with wp.ScopedCapture(device=model.device) as graph:
            substeps()
    history = []
    for _ in range(steps // 2):
        if capture:
            wp.capture_launch(graph.graph)
        else:
            substeps()
        history.append(state_0.joint_q.numpy().copy())
    return np.asarray(history)


def test_pgs_drive_matches_backward_euler_reference(test, device):
    """Reach the implicit PD drive solution below the effort limit: the force-drive row converges to backward Euler."""
    np.testing.assert_allclose(
        _slider_trajectory(_build_slider(device), 120), _backward_euler_trajectory(120), rtol=1.0e-4, atol=1.0e-5
    )


def test_pgs_drive_impulse_is_clamped_to_the_effort_limit(test, device):
    """Clamp the drive row's impulse to ``joint_effort_limit * dt``."""
    effort_limit = 5.0
    history = _slider_trajectory(_build_slider(device, effort_limit=effort_limit), 60)
    qd = np.concatenate(([0.0], history[:, 1]))
    drive_force = (MASS + ARMATURE) * np.diff(qd) / DT
    test.assertLessEqual(float(np.max(np.abs(drive_force))), effort_limit * (1.0 + 1.0e-3))
    # The drive saturates early in the step response.
    test.assertGreater(float(np.max(np.abs(drive_force))), 0.99 * effort_limit)


def test_saturated_pgs_drive_differs_from_the_augmented_drive(test, device):
    """Differ from the implicit drive when the effort limit saturates, at any iteration count.

    The augmented drive clamps the explicit drive force before the implicit solve, so the
    clamped force is divided by the drive-augmented mass. The drive row clamps its
    accumulated impulse at ``joint_effort_limit * dt``, which acts on the plain mass.
    """
    effort_limit = 5.0
    model = _build_slider(device, effort_limit=effort_limit)
    qd = {}
    for drive_mode in ("augmented", "rows"):
        solver = SolverSteel(model, solve_mode="matrix_free", drive_mode=drive_mode, iterations=100)
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, model.control(), None, DT)
        qd[drive_mode] = float(state_1.joint_qd.numpy()[0])
    augmented_mass = MASS + ARMATURE + DT * KD + DT * DT * KE
    test.assertAlmostEqual(qd["augmented"], DT * effort_limit / augmented_mass, delta=1.0e-6)
    test.assertAlmostEqual(qd["rows"], DT * effort_limit / (MASS + ARMATURE), delta=1.0e-6)
    test.assertGreater(qd["rows"] - qd["augmented"], 1.0e-3)


def test_pgs_drive_bounds_complete_reaction_under_external_load(test, device):
    """Bound the complete drive reaction by ``joint_effort_limit`` under a large external load.

    The augmented drive clamps only its explicit force, so its implicit reaction is unbounded in this scene.
    """
    dt = 0.01
    for effort_limit in (1.0, 2.0):
        for force in (-10000.0, 10000.0):
            with test.subTest(effort_limit=effort_limit, force=force):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                joint = builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=newton.Axis.X,
                    target_ke=10000.0,
                    target_kd=0.0,
                    target_pos=0.0,
                    damping=0.0,
                    armature=0.0,
                    effort_limit=effort_limit,
                )
                builder.add_articulation([joint])
                model = builder.finalize(device=device)
                solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows")
                state_0, state_1 = model.state(), model.state()
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                state_0.body_f.assign(np.array([[force, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
                solver.step(state_0, state_1, model.control(), None, dt)
                reaction = float(state_1.joint_qd.numpy()[0]) / dt - force
                test.assertLessEqual(abs(reaction), 1.05 * effort_limit)
                test.assertAlmostEqual(reaction, -np.sign(force) * effort_limit, delta=0.05)


def test_drive_rows_precede_limit_rows_and_count_against_capacity(test, device):
    """Allocate drive rows first in each world and flag the world when they do not fit."""
    model = _build_driven_chain(device, num_links=3, num_worlds=2)
    solver = SolverSteel(
        model, solve_mode="matrix_free", drive_mode="rows", enable_joint_limits=True, warn_constraint_overflow=False
    )
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    counts = solver._constraint_count.numpy()
    row_type = solver._row_type.numpy()
    np.testing.assert_array_equal(counts, [9, 9])  # 3 drive rows + 2 position-limit rows per DOF
    for world in range(2):
        test.assertTrue(np.all(row_type[world, :3] == PGS_CONSTRAINT_TYPE_JOINT_TARGET))
        test.assertTrue(np.all(row_type[world, 3:9] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))

    with test.assertWarnsRegex(UserWarning, r"need at least 6 dense rows .* articulation_constraint_max=2"):
        small = SolverSteel(
            model,
            solve_mode="matrix_free",
            drive_mode="rows",
            enable_joint_limits=True,
            articulation_constraint_max=2,
            warn_constraint_overflow=False,
        )
    small.step(model.state(), model.state(), model.control(), None, DT)
    np.testing.assert_array_equal(small._constraint_count.numpy(), [2, 2])
    test.assertTrue(np.all(small._row_type.numpy()[:, :2] == PGS_CONSTRAINT_TYPE_JOINT_TARGET))
    np.testing.assert_array_equal(small.constraint_overflow.numpy(), [True, True, False])
    with test.assertRaisesRegex(RuntimeError, "overflow|capacity"):
        small.check_constraint_capacity()


def test_undriven_dofs_get_no_drive_rows(test, device):
    """Create no drive rows for DOFs with zero target stiffness and damping."""
    model = _build_driven_chain(device, num_links=3, num_worlds=1)
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    ke[1] = 0.0
    kd[1] = 0.0
    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)
    solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", joint_limit_activation_gap=0.0)
    solver.step(model.state(), model.state(), model.control(), None, DT)
    test.assertEqual(int(solver._constraint_count.numpy()[0]), 2)
    np.testing.assert_array_equal(solver._drive_slot.numpy(), [0, -1, 1])


def test_prescale_skips_fused_driven_dofs(test, device):
    """Exclude driven DOFs from the pre-solve velocity scaling only when the fused clamp limits them."""

    def prescaled(skip_driven):
        joint_qd = wp.array([5.0, 0.5], dtype=wp.float32, device=device)
        wp.launch(
            prescale_joint_velocity_limits,
            dim=1,
            inputs=[
                wp.array([0, 2], dtype=wp.int32, device=device),
                wp.array([int(JointType.REVOLUTE)] * 2, dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([[0, 1], [0, 1]], dtype=wp.int32, device=device),
                wp.array([1.0, 1.0], dtype=wp.float32, device=device),
                wp.array([int(BodyFlags.DYNAMIC)] * 2, dtype=wp.int32, device=device),
                wp.array([0, -1], dtype=wp.int32, device=device),
                skip_driven,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[joint_qd],
            device=device,
        )
        return joint_qd.numpy()

    # Without fusion the driven DOF's overshoot scales the whole articulation.
    np.testing.assert_allclose(prescaled(0), [1.0, 0.1], rtol=1.0e-6)
    # With fusion the driven DOF is clamped in the solve and nothing is scaled.
    np.testing.assert_allclose(prescaled(1), [5.0, 0.5], rtol=1.0e-6)


def test_drive_rows_use_the_exact_unit_response(test, device):
    """Leave CFM out of drive rows: the drive and the fused clamp divide by the exact response."""
    model = _build_driven_chain(device, num_links=3, num_worlds=1)
    diags = {}
    for cfm in (0.0, 0.5):
        solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", enable_joint_limits=True, cfm=cfm)
        solver.step(model.state(), model.state(), model.control(), None, DT)
        diags[cfm] = solver._diag.numpy()[0, : int(solver._constraint_count.numpy()[0])]
    test.assertEqual(diags[0.0].size, 9)  # 3 drive rows + 2 position-limit rows per DOF
    np.testing.assert_array_equal(diags[0.5][:3], diags[0.0][:3])  # drive rows
    np.testing.assert_allclose(diags[0.5][3:] - diags[0.0][3:], 0.5, rtol=1.0e-5)  # limit rows
    # With a large CFM the PGS drive still reaches the backward-Euler reference.
    np.testing.assert_allclose(
        _slider_trajectory(_build_slider(device), 120, cfm=0.5),
        _backward_euler_trajectory(120),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_fused_clamp_lands_on_the_limit_with_large_cfm(test, device):
    """Return a driven DOF exactly to its limit regardless of cfm."""
    model = _build_arm_and_box_model_pgs_drives(device)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        drive_mode="rows",
        enable_joint_velocity_limits=True,
        cfm=0.5,
        iterations=8,
    )
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[3] = -3.0
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
    # The impact drives the arm past its 0.5 rad/s limit; the clamp leaves it at the limit.
    test.assertAlmostEqual(float(abs(state_1.joint_qd.numpy()[0])), 0.5, delta=1.0e-6)


def test_captured_drive_rows_match_eager_steps(test, device):
    """Replay a captured graph of drive, velocity-limit and contact rows identically to eager steps."""
    model = _build_arm_and_box_model_pgs_drives(device)
    kwargs = {"drive_mode": "rows", "enable_joint_velocity_limits": True, "iterations": 8}
    eager = _run_arm_and_box(model, SolverSteel(model, solve_mode="matrix_free", **kwargs), 20)
    captured = _run_arm_and_box(model, SolverSteel(model, solve_mode="matrix_free", **kwargs), 20, capture=True)
    np.testing.assert_allclose(captured, eager, atol=2.0e-6)


def test_streamed_drive_metadata_matches_resident(test, device):
    """Produce the same trajectory whether the drive-row metadata is resident or streamed."""
    model = _build_arm_and_box_model_pgs_drives(device)
    max_shared = int(getattr(model.device, "max_shared_memory_per_block", 0))
    resident_rows, streamed_rows = 32, 160
    for rows, resident in ((resident_rows, True), (streamed_rows, False)):
        test.assertEqual(
            _use_resident_mfgs_metadata(rows, 32, 7, max_shared, has_drive_rows=True, fuse_vel_limits=True), resident
        )
    kwargs = {"drive_mode": "rows", "enable_joint_velocity_limits": True, "iterations": 8}
    reference = _run_arm_and_box(
        model,
        SolverSteel(model, solve_mode="matrix_free", articulation_constraint_max=resident_rows, **kwargs),
        20,
    )
    streamed = _run_arm_and_box(
        model,
        SolverSteel(model, solve_mode="matrix_free", articulation_constraint_max=streamed_rows, **kwargs),
        20,
    )
    np.testing.assert_allclose(streamed, reference, atol=1.0e-6)


class TestSteelPGSDrives(unittest.TestCase):
    pass


for _name, _func in (
    ("test_pgs_drive_matches_backward_euler_reference", test_pgs_drive_matches_backward_euler_reference),
    ("test_pgs_drive_impulse_is_clamped_to_the_effort_limit", test_pgs_drive_impulse_is_clamped_to_the_effort_limit),
    (
        "test_saturated_pgs_drive_differs_from_the_augmented_drive",
        test_saturated_pgs_drive_differs_from_the_augmented_drive,
    ),
    (
        "test_pgs_drive_bounds_complete_reaction_under_external_load",
        test_pgs_drive_bounds_complete_reaction_under_external_load,
    ),
    (
        "test_drive_rows_precede_limit_rows_and_count_against_capacity",
        test_drive_rows_precede_limit_rows_and_count_against_capacity,
    ),
    ("test_undriven_dofs_get_no_drive_rows", test_undriven_dofs_get_no_drive_rows),
    ("test_prescale_skips_fused_driven_dofs", test_prescale_skips_fused_driven_dofs),
    ("test_drive_rows_use_the_exact_unit_response", test_drive_rows_use_the_exact_unit_response),
    ("test_fused_clamp_lands_on_the_limit_with_large_cfm", test_fused_clamp_lands_on_the_limit_with_large_cfm),
    ("test_captured_drive_rows_match_eager_steps", test_captured_drive_rows_match_eager_steps),
    ("test_streamed_drive_metadata_matches_resident", test_streamed_drive_metadata_matches_resident),
):
    add_function_test(TestSteelPGSDrives, _name, _func, devices=devices)


_HAS_DRIVE_ROWS = "drive_mode" in inspect.signature(SolverSteel.__init__).parameters


_DT = 1.0 / 120.0


def _driven_lever_on_ground(device):
    """A horizontal lever held by a stiff revolute drive whose 1 kg foot rests 1 mm deep in the ground.

    Drive and contact share the weight. The contact's position bias pushes the foot out
    during the position solve; the velocity-only iterations drop that bias, so the contact
    impulse changes there and an active drive row reacts to it.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity()))
    # A foot at the far end of the lever touches the ground; the hinge itself has no shape.
    builder.add_shape_box(
        link,
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.05,
        hy=0.05,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
    )
    joint = builder.add_joint_revolute(
        -1,
        link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity()),
        target_ke=1000.0,
        target_kd=50.0,
        target_pos=0.0,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _one_step(device, **solver_kwargs):
    model = _driven_lever_on_ground(device)
    solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", iterations=4, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, _DT)
    drive_slot = int(solver._drive_slot.numpy()[0])
    count = int(solver._constraint_count.numpy()[0])
    row_type = solver._row_type.numpy()[0, :count]
    impulses = solver._impulses.numpy()[0, :count]
    contact_impulse = float(impulses[row_type == PGS_CONSTRAINT_TYPE_CONTACT].sum())
    return float(impulses[drive_slot]), contact_impulse


@unittest.skipUnless(_HAS_DRIVE_ROWS, "PGS joint-drive rows (drive_mode='rows') are not available")
def test_frozen_drive_rows_keep_the_position_solve_impulse(test, device):
    """Keep the position solve's drive impulse under ``"freeze"`` and keep solving the drive rows under ``"active"``."""
    position_drive, position_contact = _one_step(device, velocity_iterations=0)
    frozen_drive, frozen_contact = _one_step(device, velocity_iterations=8, velocity_drive_mode="freeze")
    active_drive, _active_contact = _one_step(device, velocity_iterations=8, velocity_drive_mode="active")

    # The velocity-only iterations change the contact impulse (its position bias is dropped).
    test.assertGreater(abs(frozen_contact - position_contact), 1.0e-4 * abs(position_contact))
    # A frozen drive row keeps exactly its position-solve impulse; an active one reacts.
    test.assertEqual(frozen_drive, position_drive)
    test.assertGreater(abs(active_drive - position_drive), 1.0e-4 * abs(position_drive))


class TestSteelVelocityDriveMode(unittest.TestCase):
    pass


add_function_test(
    TestSteelVelocityDriveMode,
    "test_frozen_drive_rows_keep_the_position_solve_impulse",
    test_frozen_drive_rows_keep_the_position_solve_impulse,
    devices=get_cuda_test_devices(),
)


def _build_slider_augmented_drives(device, *, effort_limit=1.0e6, damping=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=MASS, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=newton.Axis.X,
        target_ke=KE,
        target_kd=KD,
        target_pos=TARGET_POS,
        target_vel=TARGET_VEL,
        armature=ARMATURE,
        effort_limit=effort_limit,
        damping=damping,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _reference_trajectory(steps, *, effort_limit=1.0e6, damping=0.0):
    """Backward-Euler PD drive: ``(m + a + dt kd + dt^2 ke) qdd = clamp(u0) - c qd``."""
    q = 0.0
    qd = 0.0
    history = []
    effective_mass = MASS + ARMATURE + DT * KD + DT * DT * KE
    for _ in range(steps):
        u0 = -(KE * (q - TARGET_POS + DT * qd) + KD * (qd - TARGET_VEL))
        u0 = float(np.clip(u0, -effort_limit, effort_limit))
        qdd = (u0 - damping * qd) / effective_mass
        qd = qd + DT * qdd
        q = q + DT * qd
        history.append((q, qd))
    return np.asarray(history)


def _solver_trajectory(model, steps, solve_mode):
    solver = SolverSteel(model, solve_mode=solve_mode)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    history = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        history.append((float(state_0.joint_q.numpy()[0]), float(state_0.joint_qd.numpy()[0])))
    return np.asarray(history)


def test_implicit_drive_matches_backward_euler_reference(test, device, solve_mode="matrix_free"):
    """Integrate the PD drive implicitly through the augmented mass matrix."""
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider_augmented_drives(device), 120, solve_mode),
        _reference_trajectory(120),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_drive_force_is_clamped_to_the_effort_limit(test, device, solve_mode="matrix_free"):
    """Clamp the explicit drive force to the joint effort limit before the implicit solve."""
    reference = _reference_trajectory(120, effort_limit=5.0)
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider_augmented_drives(device, effort_limit=5.0), 120, solve_mode),
        reference,
        rtol=1.0e-4,
        atol=1.0e-5,
    )
    test.assertGreater(float(np.abs(reference - _reference_trajectory(120)).max()), 1.0e-2, "the clamp must be active")


def test_passive_joint_damping_is_applied(test, device, solve_mode="matrix_free"):
    """Apply Model.joint_damping as a passive joint force."""
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider_augmented_drives(device, damping=3.0), 120, solve_mode),
        _reference_trajectory(120, damping=3.0),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


class TestSteelAugmentedDrives(unittest.TestCase):
    pass


for _name, _func in (
    ("test_implicit_drive_matches_backward_euler_reference", test_implicit_drive_matches_backward_euler_reference),
    ("test_drive_force_is_clamped_to_the_effort_limit", test_drive_force_is_clamped_to_the_effort_limit),
    ("test_passive_joint_damping_is_applied", test_passive_joint_damping_is_applied),
):
    add_function_test(TestSteelAugmentedDrives, _name, _func, devices=devices)
    add_function_test(TestSteelAugmentedDrives, f"{_name}_split", _func, devices=split_devices, solve_mode="split")


if __name__ == "__main__":
    unittest.main()
