# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint limits, velocity limits, joint friction and joint drives of SolverSteel."""

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
    ROW_SLOTS_ATOMIC,
    ROW_SLOTS_COUNT,
    ROW_SLOTS_WRITE,
    allocate_joint_velocity_limit_slots,
    allocate_physx_drive_slots,
    allocate_rigid_velocity_limit_slots,
    build_joint_limit_rows,
    prescale_joint_velocity_limits,
)
from newton._src.solvers.steel.solver_steel import _get_joint_limit_warp_kernel, _use_resident_mfgs_metadata
from newton.solvers import SolverBase, SolverSteel
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_cuda_test_devices, get_test_devices


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
        wp.launch(
            build_joint_limit_rows,
            dim=1,
            inputs=[*inputs, 1],
            outputs=[*outputs, ROW_SLOTS_ATOMIC, wp.zeros(1, dtype=wp.int32, device=device)],
            device=device,
        )
    else:
        kernel = _get_joint_limit_warp_kernel(1, wp.get_device(device).arch, 1)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[1, *inputs],
            outputs=[*outputs, ROW_SLOTS_ATOMIC, wp.zeros(1, dtype=wp.int32, device=device)],
            block_dim=32,
            device=device,
        )
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
        propagation_count = int(solver._propagation_constraint_count.numpy()[0])
        samples.append(
            {
                "dense_count": dense_count,
                "row_type": solver._row_type.numpy()[0, :dense_count].copy(),
                "impulses": solver._impulses.numpy()[0, :dense_count].copy(),
                "mf_row_type": solver._mf_row_type.numpy()[0, :mf_count].copy(),
                "mf_impulses": solver._mf_impulses.numpy()[0, :mf_count].copy(),
                "propagation_count": propagation_count,
                "propagation_impulses": solver._propagation_impulses.numpy()[0, :propagation_count].copy(),
                "joint_q": state_1.joint_q.numpy().copy(),
                "joint_qd": state_1.joint_qd.numpy().copy(),
            }
        )
        state_0, state_1 = state_1, state_0
    return samples


def test_propagation_solves_every_row_family(test, device):
    """Solve every row family in the propagation schedule and route every contact to propagation rows.

    Each propagation iteration solves the dense joint-limit rows, the contact rows, the
    propagation rows and their tree propagation, then the velocity-limit rows. The scene's
    articulated link sits on its limit against the ground and its free body has velocity limits.
    """
    for response in ("propagation", "propagation_colored"):
        for iterations in (1, 8):
            with test.subTest(response=response, iterations=iterations):
                samples = _run_layout_trajectory(device, response, iterations)
                for step, sample in enumerate(samples):
                    types = sample["row_type"]
                    # Dense rows hold only the joint rows; every contact touching the link is a propagation row.
                    test.assertGreater(int(np.sum(types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)), 0, f"step {step}")
                    test.assertTrue(
                        np.all(
                            (types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
                            | (types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
                        )
                    )
                    test.assertGreater(sample["propagation_count"], 0)
                    test.assertGreater(
                        int(np.sum(sample["mf_row_type"] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)), 0
                    )
                    # The free body's contacts are propagation rows too.
                    test.assertEqual(int(np.sum(sample["mf_row_type"] == PGS_CONSTRAINT_TYPE_CONTACT)), 0)
                limit_rows = np.flatnonzero(samples[0]["row_type"] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
                test.assertGreater(float(np.max(np.abs(samples[0]["impulses"][limit_rows]))), 0.0)
                # The propagation and velocity-limit solves apply impulses too.
                test.assertGreater(max(float(np.max(np.abs(s["propagation_impulses"]))) for s in samples), 0.0)
                velocity_limit_impulses = [
                    s["mf_impulses"][s["mf_row_type"] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT] for s in samples
                ]
                test.assertGreater(max(float(np.max(np.abs(i))) for i in velocity_limit_impulses), 0.0)


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
    ("test_propagation_solves_every_row_family", test_propagation_solves_every_row_family),
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
            outputs=[*outputs, ROW_SLOTS_ATOMIC, wp.zeros(1, dtype=wp.int32, device=device)],
            device=device,
        )
    else:
        warps_per_block = 4
        kernel = _get_joint_limit_warp_kernel(size, wp.get_device(device).arch, warps_per_block)
        wp.launch_tiled(
            kernel,
            dim=[(articulation_count + warps_per_block - 1) // warps_per_block],
            inputs=[articulation_count, *inputs],
            outputs=[*outputs, ROW_SLOTS_ATOMIC, wp.zeros(1, dtype=wp.int32, device=device)],
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
            wp.array([1], dtype=wp.int32, device=device),
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
        outputs=[
            velocity_limit_slot,
            velocity_limit_sign,
            world_slot_counter,
            ROW_SLOTS_ATOMIC,
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
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
            wp.array([wp.quat_identity()], dtype=wp.quat, device=device),
            fraction,
            64,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[
            rigid_velocity_limit_slot,
            rigid_velocity_limit_sign,
            mf_slot_counter,
            ROW_SLOTS_ATOMIC,
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
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


def test_count_pass_assigns_no_drive_slot_above_the_sentinel(test, device):
    """Leave a drive slot unassigned on the COUNT pass even when the row capacity exceeds the pass's sentinel."""
    max_constraints = (1 << 30) + 1

    def ints(values):
        return wp.array(values, dtype=wp.int32, device=device)

    # (drive_slot, world_slot_counter, element_row_cursor) after one driven prismatic DOF.
    expected = {
        int(ROW_SLOTS_ATOMIC): ([0], [1], [0]),
        int(ROW_SLOTS_COUNT): ([-1], [0], [1]),
        int(ROW_SLOTS_WRITE): ([0], [0], [1]),
    }
    for mode, result in expected.items():
        drive_slot, world_slot_counter, element_row_cursor = ints([-7]), ints([0]), ints([0])
        wp.launch(
            allocate_physx_drive_slots,
            dim=1,
            inputs=[
                ints([0, 1]),
                ints([1]),
                ints([0]),
                ints([1]),
                ints([int(JointType.PRISMATIC)]),
                ints([0]),
                wp.array([[1, 0]], dtype=wp.int32, device=device),
                wp.array([1.0], dtype=wp.float32, device=device),
                wp.array([0.0], dtype=wp.float32, device=device),
                ints([0]),
                max_constraints,
            ],
            outputs=[drive_slot, world_slot_counter, mode, element_row_cursor],
            device=device,
        )
        with test.subTest(mode=mode):
            test.assertEqual(
                (drive_slot.numpy().tolist(), world_slot_counter.numpy().tolist(), element_row_cursor.numpy().tolist()),
                result,
            )


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

    with test.assertWarnsRegex(
        newton.exceptions.NewtonWarning, r"need at least 6 dense rows .* articulation_constraint_max=2"
    ) as caught:
        small = SolverSteel(
            model,
            solve_mode="matrix_free",
            drive_mode="rows",
            enable_joint_limits=True,
            articulation_constraint_max=2,
            warn_constraint_overflow=False,
        )
    test.assertNotIn("friction", str(caught.warning))
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
                wp.array([2], dtype=wp.int32, device=device),
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
add_function_test(
    TestSteelPGSDrives,
    "test_count_pass_assigns_no_drive_slot_above_the_sentinel",
    test_count_pass_assigns_no_drive_slot_above_the_sentinel,
    devices=split_devices,
)


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


DT_JOINT_FRICTION = 0.01


INERTIA = 0.5


FRICTION = 1.0


@unittest.skipUnless(wp.is_cuda_available(), "matrix-free joint friction requires CUDA")
class TestSteelJointFriction(unittest.TestCase):
    def test_sticks_below_breakaway_in_both_directions(self):
        """Hold hinge and slider DOFs at rest while the applied effort is below the friction."""
        efforts = (0.6 * FRICTION, -0.6 * FRICTION, 0.999 * FRICTION)
        for joint in ("revolute", "prismatic"):
            with self.subTest(joint=joint):
                qd = _run_friction(_worlds_model(joint, len(efforts)), effort=efforts, steps=50, dofs=len(efforts))
                np.testing.assert_allclose(qd, 0.0, atol=1.0e-6)

    def test_breakaway_accelerates_by_net_effort(self):
        """Accelerate at (effort - friction) / inertia once the effort exceeds the friction."""
        for joint, inertia in (("revolute", INERTIA), ("prismatic", 1.0)):
            with self.subTest(joint=joint):
                qd = _run_friction(_worlds_model(joint, 2), effort=(2.5, -2.5), steps=20, dofs=2)
                ramp = (2.5 - FRICTION) / inertia * DT_JOINT_FRICTION * np.arange(1, 21)
                np.testing.assert_allclose(qd, np.stack((ramp, -ramp), axis=1), rtol=1.0e-5, atol=1.0e-6)

    def test_sliding_decelerates_and_stops_without_chatter(self):
        """Decelerate at friction / inertia, stop at the predicted step and stay at rest."""
        qd = _run_friction(_worlds_model("revolute", 2), qd0=(1.0, -1.0), steps=80, dofs=2)
        stop_step = round(INERTIA / FRICTION / DT_JOINT_FRICTION)
        speed = np.maximum(1.0 - FRICTION / INERTIA * DT_JOINT_FRICTION * np.arange(1, 81), 0.0)
        np.testing.assert_allclose(qd, np.stack((speed, -speed), axis=1), atol=1.0e-5)
        np.testing.assert_allclose(qd[stop_step:], 0.0, atol=1.0e-6)
        for world, sign in ((0, 1.0), (1, -1.0)):
            moving = qd[:, world][np.abs(qd[:, world]) > 1.0e-6]
            self.assertTrue(np.all(np.sign(moving) == sign))

    def test_pendulum_dissipates_the_friction_work(self):
        """Lose mechanical energy at the rate of the friction torque times the swept angle."""
        mass, offset, inertia, friction, dt = 1.0, 0.3, 0.01, 0.1, 0.002
        losses = {}
        for enabled in (False, True):
            model = _pendulum_model(friction=friction, mass=mass, offset=offset, inertia=inertia)
            solver = _friction_solver(model, enable_joint_friction=enabled)
            states, control = [model.state(), model.state()], model.control()
            states[0].joint_q.assign([1.0])
            newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
            q, qd = _advance_friction(solver, states, control, 300, dt)
            q = np.concatenate(([1.0], q))
            qd = np.concatenate(([0.0], qd))
            # Height of the center of mass below the pivot for a hinge about +y with the COM on +x.
            energy = 0.5 * (inertia + mass * offset**2) * qd**2 - mass * 9.81 * offset * np.sin(q)
            self.assertGreater(float(np.ptp(q)), 1.0)
            losses[enabled] = (float(energy[0] - energy[-1]), friction * float(np.sum(np.abs(np.diff(q)))))
        self.assertLess(abs(losses[False][0]), 0.01 * losses[True][1])
        np.testing.assert_allclose(losses[True][0], losses[True][1], rtol=0.05)

    def test_substeps_preserve_physical_result(self):
        """Scale the impulse bound with dt so substepping keeps the same physical trajectory."""
        model = _worlds_model("revolute", 2)
        for substeps in (1, 4):
            with self.subTest(substeps=substeps):
                dt = DT_JOINT_FRICTION / substeps
                # World 0 breaks away under effort, world 1 slides to rest.
                qd = _run_friction(model, effort=(2.5, 0.0), qd0=(0.0, 1.0), steps=60 * substeps, dt=dt, dofs=2)
                self.assertAlmostEqual(float(qd[20 * substeps - 1, 0]), 1.5 / INERTIA * 0.2, delta=1.0e-5)
                stop = int(np.argmax(np.abs(qd[:, 1]) < 1.0e-6)) + 1
                self.assertAlmostEqual(stop * dt, 0.5, delta=0.5 * dt)

    def test_d6_axes_are_independent_boxes(self):
        """Apply a separate box to each D6 axis, as MuJoCo does per DOF."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA))
        cfg = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            -1,
            body,
            linear_axes=[cfg(axis=(1.0, 0.0, 0.0), friction=FRICTION, armature=0.0)],
            angular_axes=[cfg(axis=(0.0, 0.0, 1.0), friction=2.0 * FRICTION, armature=0.0)],
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        qd = _run_friction(model, effort=(1.5, 1.5), steps=10, dofs=2)
        np.testing.assert_allclose(qd[-1], [0.5 * DT_JOINT_FRICTION * 10, 0.0], atol=1.0e-5)

    def test_ball_axes_match_mujoco_frictionloss(self):
        """Box each BALL DOF separately, matching analytic sticking and SolverMuJoCo's frictionloss when sliding."""
        efforts = np.array([[0.6, -0.5, 0.9], [2.5, 0.0, 0.0], [0.0, -2.5, 0.3], [2.0, 1.8, 0.0]], dtype=np.float32)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in efforts:
            builder.add_world(_root_joint_builder("ball", friction=FRICTION))
        model = builder.finalize()
        qd = _run_friction(model, effort=efforts.ravel(), steps=20, dofs=efforts.size).reshape(20, -1, 3)
        sliding = np.abs(efforts) > FRICTION
        np.testing.assert_allclose(qd[:, ~sliding], 0.0, atol=1.0e-6)
        ramp = (2.5 - FRICTION) / INERTIA * DT_JOINT_FRICTION * np.arange(1, 21)
        np.testing.assert_allclose(qd[:, 1, 0], ramp, atol=1.0e-5)
        try:
            reference = newton.solvers.SolverMuJoCo(model, iterations=100, ls_iterations=50, cone="pyramidal")
        except ImportError as error:
            self.skipTest(f"MuJoCo is not installed: {error}")
        states, control = [model.state(), model.state()], model.control()
        control.joint_f.assign(efforts.ravel())
        qd_mujoco = _advance_friction(reference, states, control, 20)[1].reshape(20, -1, 3)
        # MuJoCo's soft friction creeps below breakaway, so compare the sliding DOFs only.
        np.testing.assert_allclose(qd[:, sliding], qd_mujoco[:, sliding], atol=1.0e-4)

    def test_locked_child_moves_with_composite_inertia(self):
        """Lock a stiff-friction child joint so the chain accelerates with its composite inertia."""
        length, mass = 0.4, 2.0
        model = _two_link_model(length=length, mass=mass, friction=(FRICTION, 50.0))
        qd = _run_friction(model, effort=(3.0, 0.0), steps=10, dofs=2)
        composite = 2.0 * INERTIA + mass * length**2
        np.testing.assert_allclose(
            qd[:, 0], (3.0 - FRICTION) / composite * DT_JOINT_FRICTION * np.arange(1, 11), rtol=1.0e-4
        )
        np.testing.assert_allclose(qd[:, 1], 0.0, atol=1.0e-5)

    def test_drive_settles_inside_friction_deadband(self):
        """Stop a PD-driven hinge where the drive effort falls inside the friction box."""
        ke, kd, target = 10.0, 2.0, 1.0
        for drive_mode in ("augmented", "rows"):
            with self.subTest(drive_mode=drive_mode):
                model = _single_dof_model("revolute", ke=ke, kd=kd)
                solver = _friction_solver(model, drive_mode=drive_mode, iterations=32)
                states, control = [model.state(), model.state()], model.control()
                control.joint_target_q.assign([target])
                q, qd = _advance_friction(solver, states, control, 400)
                self.assertLess(abs(float(qd[-1])), 1.0e-5)
                self.assertLessEqual(abs(float(q[-1]) - target), FRICTION / ke + 1.0e-3)
                self.assertGreater(abs(float(q[-1]) - target), 1.0e-2)
                reference = _friction_solver(model, drive_mode=drive_mode, iterations=32, enable_joint_friction=False)
                states = [model.state(), model.state()]
                q_free, _ = _advance_friction(reference, states, control, 400)
                self.assertAlmostEqual(float(q_free[-1]), target, delta=1.0e-3)

    def test_slider_stops_at_position_limit(self):
        """Combine friction deceleration with a position limit without violating the limit."""
        model = _single_dof_model("prismatic", limit=(-1.0, 0.1))
        solver = _friction_solver(model, enable_joint_limits=True)
        states, control = [model.state(), model.state()], model.control()
        states[0].joint_qd.assign([2.0])
        q, qd = _advance_friction(solver, states, control, 60)
        self.assertLess(float(np.max(q)), 0.1 + 1.0e-3)
        np.testing.assert_allclose(qd[-10:], 0.0, atol=1.0e-5)

    def test_velocity_limit_keeps_last_word(self):
        """Saturate breakaway acceleration at the joint velocity limit."""
        model = _single_dof_model("revolute", velocity_limit=1.0)
        for schedule in ("interleaved", "phased"):
            with self.subTest(schedule=schedule):
                solver = _friction_solver(model, enable_joint_velocity_limits=True, row_schedule=schedule)
                states, control = [model.state(), model.state()], model.control()
                control.joint_f.assign([10.0])
                _, qd = _advance_friction(solver, states, control, 20)
                self.assertLessEqual(float(np.max(qd)), 1.0 + 1.0e-5)
                self.assertAlmostEqual(float(qd[-1]), 1.0, delta=1.0e-5)

    def test_friction_and_ground_contact(self):
        """Hold a vertical slider against gravity, or let it fall onto the ground and rest."""
        for friction, falls in ((15.0, False), (5.0, True)):
            with self.subTest(friction=friction):
                model = _vertical_slider_model(mass=1.0, half=0.1, friction=friction)
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
                contacts = pipeline.contacts()
                solver = _friction_solver(model, friction_anchor_beta=0.0, iterations=24)
                states, control = [model.state(), model.state()], model.control()
                newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
                for _ in range(150):
                    pipeline.collide(states[0], contacts)
                    solver.step(states[0], states[1], control, contacts, DT_JOINT_FRICTION)
                    states.reverse()
                z = float(states[0].joint_q.numpy()[0])
                if falls:
                    self.assertAlmostEqual(z, -0.4, delta=5.0e-3)
                else:
                    self.assertAlmostEqual(z, 0.0, delta=1.0e-5)
                self.assertLess(abs(float(states[0].joint_qd.numpy()[0])), 1.0e-3)

    def test_heterogeneous_worlds_match_isolated_worlds(self):
        """Solve per-world friction rows in worlds with different DOF counts and coefficients."""
        single = _single_dof_model("revolute")
        chain = _two_link_model(length=0.4, mass=2.0, friction=(FRICTION, 0.2))
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(_single_dof_builder("revolute"))
        builder.add_world(_two_link_builder(length=0.4, mass=2.0, friction=(FRICTION, 0.2)))
        mixed = builder.finalize()
        effort = (2.5, 3.0, 0.1)
        qd_mixed = _run_friction(mixed, effort=effort, steps=20, dofs=3)
        np.testing.assert_allclose(qd_mixed[:, 0], _run_friction(single, effort=effort[0], steps=20), atol=1.0e-6)
        np.testing.assert_allclose(
            qd_mixed[:, 1:], _run_friction(chain, effort=effort[1:], steps=20, dofs=2), atol=1.0e-6
        )

    def test_zero_or_disabled_friction_matches_disabled_bitwise(self):
        """Keep zero coefficients, and disabled friction with nonzero coefficients, bit-identical to no friction."""
        trajectories = []
        for friction, enabled in (((0.0, 0.0), False), ((0.0, 0.0), True), ((FRICTION, FRICTION), False)):
            model = _two_link_model(length=0.4, mass=2.0, friction=friction, ke=20.0, kd=1.0)
            solver = _friction_solver(model, enable_joint_friction=enabled, enable_joint_limits=True)
            states, control = [model.state(), model.state()], model.control()
            states[0].joint_qd.assign([1.0, -2.0])
            control.joint_target_q.assign([0.3, -0.2])
            trajectories.append(_advance_friction(solver, states, control, 30))
        for trajectory in trajectories[1:]:
            np.testing.assert_array_equal(trajectory[0], trajectories[0][0])
            np.testing.assert_array_equal(trajectory[1], trajectories[0][1])

    def test_runtime_edits_take_effect(self):
        """Apply in-place edits and array replacement after notification."""
        model = _single_dof_model("revolute", friction=0.0)
        solver = _friction_solver(model)
        states, control = [model.state(), model.state()], model.control()
        control.joint_f.assign([0.5])
        _, qd = _advance_friction(solver, states, control, 10)
        self.assertAlmostEqual(float(qd[-1]), 0.5 / INERTIA * 0.1, delta=1.0e-5)

        model.joint_friction.assign([FRICTION])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _, qd = _advance_friction(solver, states, control, 20)
        np.testing.assert_allclose(qd[-5:], 0.0, atol=1.0e-6)

        control.joint_f.assign([1.5])
        model.joint_friction = wp.array([2.0], dtype=wp.float32, device=model.device)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        _, qd = _advance_friction(solver, states, control, 5)
        np.testing.assert_allclose(qd, 0.0, atol=1.0e-6)

    def test_strided_coefficients_are_read_by_stride(self):
        """Read strided coefficient views at construction and after notified replacement."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            builder.add_world(_single_dof_builder("revolute", friction=0.0))
        expected = np.array([1.5 / INERTIA * DT_JOINT_FRICTION, 0.5 / INERTIA * DT_JOINT_FRICTION], dtype=np.float32)
        for strided in (False, True):
            for when in ("construction", "replacement"):
                with self.subTest(strided=strided, when=when):
                    model = builder.finalize()
                    storage = wp.array([1.0, 99.0, 2.0, 99.0], dtype=wp.float32, device=model.device)
                    coefficients = (
                        storage[::2] if strided else wp.array([1.0, 2.0], dtype=wp.float32, device=model.device)
                    )
                    if when == "construction":
                        model.joint_friction = coefficients
                    solver = _friction_solver(model)
                    if when == "replacement":
                        model.joint_friction = coefficients
                        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                    states, control = [model.state(), model.state()], model.control()
                    control.joint_f.fill_(2.5)
                    _, qd = _advance_friction(solver, states, control, 1)
                    np.testing.assert_allclose(qd, expected, atol=1.0e-6)

    def test_graph_replay_matches_eager_and_reads_edits(self):
        """Replay captured steps bit-identically and read in-place coefficient edits."""
        model = _single_dof_model("revolute")
        eager = _run_friction(model, effort=2.5, steps=16)

        solver = _friction_solver(model)
        states, control = [model.state(), model.state()], model.control()
        # Friction holds the hinge exactly at rest while warming up kernels outside capture.
        _advance_friction(solver, states, control, 2)
        control.joint_f.assign([2.5])
        with wp.ScopedCapture(device=model.device) as capture:
            for _ in range(2):
                solver.step(states[0], states[1], control, None, DT_JOINT_FRICTION)
                solver.step(states[1], states[0], control, None, DT_JOINT_FRICTION)
        replayed = []
        for _ in range(4):
            wp.capture_launch(capture.graph)
            replayed.append(states[0].joint_qd.numpy()[0])
        np.testing.assert_array_equal(np.array(replayed, dtype=np.float32), eager[3::4])

        model.joint_friction.assign([3.0])
        before = float(states[0].joint_qd.numpy()[0])
        wp.capture_launch(capture.graph)
        self.assertAlmostEqual(
            float(states[0].joint_qd.numpy()[0]), before - 0.5 / INERTIA * 4 * DT_JOINT_FRICTION, delta=1.0e-5
        )

    def test_invalid_coefficients_build_no_rows(self):
        """Step an invalid coefficient written without a notification as frictionless and report it."""
        frictionless = _run_friction(_single_dof_model("revolute", friction=0.0), effort=2.5, steps=5)
        for value in (float("inf"), float("nan"), -1.0):
            with self.subTest(value=value):
                model = _single_dof_model("revolute", friction=0.0)
                solver = _friction_solver(model)
                states, control = [model.state(), model.state()], model.control()
                control.joint_f.assign([2.5])
                model.joint_friction.assign([value])
                _, qd = _advance_friction(solver, states, control, 5)
                np.testing.assert_array_equal(qd, frictionless)
                with self.assertRaisesRegex(RuntimeError, r"joint_friction was invalid at a step in worlds \[-1\]"):
                    solver.check_constraint_capacity()

    def test_finite_coefficient_with_unrepresentable_bound_still_sticks(self):
        """Keep the row of a finite coefficient whose impulse bound overflows to infinity."""
        model = _single_dof_model("revolute", friction=float(np.finfo(np.float32).max))
        np.testing.assert_array_equal(_run_friction(model, effort=2.5, steps=3, dt=2.0), 0.0)

    def test_captured_notify_replay_matches_eager(self):
        """Replay per-world friction writes, notify and steps bit-identically to the eager sequence."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            builder.add_world(_single_dof_builder("revolute", friction=0.0))
        rng = np.random.default_rng(4)
        coefficients = rng.uniform(0.0, 3.0, (4, 2)).astype(np.float32)
        runs = []
        for captured in (False, True):
            model = builder.finalize()
            solver = _friction_solver(model)
            states, control = [model.state(), model.state()], model.control()
            control.joint_f.assign([2.0, -2.0])
            states[0].joint_qd.assign([0.5, -0.5])
            _advance_friction(solver, states, control, 2)

            def episode(solver=solver, states=states, control=control):
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                for _ in range(2):
                    solver.step(states[0], states[1], control, None, DT_JOINT_FRICTION)
                    solver.step(states[1], states[0], control, None, DT_JOINT_FRICTION)

            if captured:
                with wp.ScopedCapture(device=model.device) as capture:
                    episode()
            qd = []
            for row in coefficients:
                model.joint_friction.assign(row)
                if captured:
                    wp.capture_launch(capture.graph)
                else:
                    episode()
                qd.append(states[0].joint_qd.numpy().copy())
            solver.check_constraint_capacity()
            runs.append(np.array(qd))
        np.testing.assert_array_equal(runs[1], runs[0])

    def test_reset_starts_from_cold_rows(self):
        """Reproduce a fresh trajectory after a full or masked reset and state rewrite."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            builder.add_world(_single_dof_builder("revolute"))
        model = builder.finalize()
        fresh = _run_friction(model, qd0=(1.0, -1.0), effort=(0.2, -0.2), steps=20, dofs=2)
        for mask in (None, [True, True, False]):
            with self.subTest(mask=mask):
                solver = _friction_solver(model)
                states, control = [model.state(), model.state()], model.control()
                control.joint_f.assign([0.2, -0.2])
                states[0].joint_qd.assign([0.3, 0.7])
                _advance_friction(solver, states, control, 7)
                states[0].joint_q.zero_()
                states[0].joint_qd.assign([1.0, -1.0])
                world_mask = None if mask is None else wp.array(mask, dtype=wp.bool, device=model.device)
                solver.reset(states[0], world_mask=world_mask)
                _, qd = _advance_friction(solver, states, control, 20)
                np.testing.assert_array_equal(qd, fresh)

    def test_dense_row_overflow_is_reported(self):
        """Warn at construction when friction rows exceed the capacity in every step, and flag the worlds."""
        model = _two_link_model(length=0.4, mass=2.0, friction=(FRICTION, FRICTION))
        with self.assertWarnsRegex(
            newton.exceptions.NewtonWarning, r"need at least 2 dense rows .* friction"
        ) as caught:
            solver = _friction_solver(model, articulation_constraint_max=1, warn_constraint_overflow=False)
        self.assertNotIn("limit", str(caught.warning))
        # Limit and friction rows together name both causes and both remedies.
        limited = _single_dof_model("revolute", limit=(-1.0, 1.0))
        with self.assertWarnsRegex(
            newton.exceptions.NewtonWarning,
            r"3 dense rows .* position limits and positive joint friction .* joint_limit_activation_gap"
            r".* set joint_friction to zero",
        ):
            _friction_solver(
                limited, enable_joint_limits=True, articulation_constraint_max=2, warn_constraint_overflow=False
            )
        states, control = [model.state(), model.state()], model.control()
        _advance_friction(solver, states, control, 1)
        self.assertTrue(bool(solver.constraint_overflow.numpy()[0]))
        with self.assertRaisesRegex(RuntimeError, "dropped contacts or constraint rows"):
            solver.check_constraint_capacity()

    def test_sleeping_island_wakes_on_friction_edit(self):
        """Sleep a friction-held pendulum and wake it when its friction is lowered."""
        model = _pendulum_model(friction=5.0, fixed_base=True)
        solver = _friction_solver(model, enable_sleeping=True, sleep_quiet_time=0.05)
        states, control = [model.state(), model.state()], model.control()
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        _advance_friction(solver, states, control, 30)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        model.joint_friction.assign([0.0])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _, qd = _advance_friction(solver, states, control, 3)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])
        self.assertGreater(abs(float(qd[-1])), 1.0e-2)

    def test_schedules_routes_and_velocity_pass_agree(self):
        """Give the same analytic result on every supported schedule and contact route."""
        model = _single_dof_model("revolute")
        expected = (2.5 - FRICTION) / INERTIA * DT_JOINT_FRICTION * 10
        options = (
            {"row_schedule": "contact_then_internal"},
            {"row_schedule": "phased"},
            {"velocity_iterations": 2},
            {"contact_warmstart": True},
            {"drive_mode": "rows"},
            {"articulated_contact_response": "propagation"},
            {"articulated_contact_response": "propagation_colored"},
        )
        for kwargs in options:
            with self.subTest(**kwargs):
                solver = _friction_solver(model, **kwargs)
                if kwargs.get("articulated_contact_response") == "propagation_colored":
                    self.assertTrue(solver._propagation_colored)
                states, control = [model.state(), model.state()], model.control()
                control.joint_f.assign([0.5])
                _, stick = _advance_friction(solver, states, control, 10)
                np.testing.assert_allclose(stick, 0.0, atol=1.0e-6)
                # Sticking leaves the hinge at rest, so the slide starts from the same state as a fresh run.
                control.joint_f.assign([2.5])
                _, slide = _advance_friction(solver, states, control, 10)
                self.assertAlmostEqual(float(slide[-1]), expected, delta=1.0e-5)

    def test_local_internal_fast_path_is_not_selected(self):
        """Keep enabled friction off the articulation-local solves.

        The selection tests of the paired factor-coordinate, sparse-diagonal and sparse-factor solves cover the others.
        """
        model = _arm_and_box_model()
        self.assertTrue(_friction_solver(model, enable_joint_friction=False)._local_internal_fast_path)
        self.assertFalse(_friction_solver(model)._local_internal_fast_path)

    def test_rejects_unsupported_configurations(self):
        """Raise for joint types, values and solver options the friction rows do not support."""
        with self.assertRaisesRegex(NotImplementedError, "matrix_free"):
            SolverSteel(_single_dof_model("revolute"), solve_mode="split", enable_joint_friction=True)
        # Point friction keeps contact_compliance from warning about friction patches first.
        for option, extra in (
            ("enable_bilateral_preelimination", {}),
            ("contact_compliance", {"friction_anchor_beta": 0.0}),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(NotImplementedError, option):
                _friction_solver(_single_dof_model("revolute"), **{option: True}, **extra)
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and non-negative"):
                _friction_solver(_single_dof_model("revolute", friction=value))
        with self.assertRaisesRegex(ValueError, r"joint_friction is nonzero on joint 0 \(FREE\)"):
            _friction_solver(_root_joint_model("free", friction=FRICTION))
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        links = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA)) for _ in range(2)]
        builder.add_articulation([builder.add_joint_revolute(-1, link, axis=newton.Axis.Z) for link in links])
        builder.add_joint_revolute(links[0], links[1], axis=newton.Axis.Z, friction=FRICTION)
        with self.assertRaisesRegex(ValueError, r"joint_friction is nonzero on joint 2 \(loop-closing joint\)"):
            _friction_solver(builder.finalize())
        # Joint-friction rows take a fourth row-type bit: at most 2**27 - 1 rows, rejected before allocation.
        with self.assertRaisesRegex(ValueError, "articulation_constraint_max must be at most 134217727"):
            _friction_solver(_single_dof_model("revolute"), articulation_constraint_max=134_217_728)

    def test_eager_notify_raises_after_applying_updates(self):
        """Raise from an eager notify after the other updates, and accept the next valid notify."""
        model = _root_joint_model("free", friction=0.0)
        solver = _friction_solver(model)
        model.joint_armature.fill_(0.3)
        model.joint_friction.fill_(FRICTION)
        with self.assertRaisesRegex(ValueError, "FREE"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertTrue(np.all(solver._joint_armature_device.numpy() == np.float32(0.3)))
        model.joint_friction.zero_()
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def test_captured_steps_flag_invalid_coefficients_until_repaired_and_reset(self):
        """Flag each invalid coefficient's world on replay, re-flag it after a reset until repaired, and reset per world."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(_single_dof_builder("revolute", friction=0.0))
        builder.add_world(_root_joint_builder("free", friction=0.0))
        builder.add_world(_single_dof_builder("revolute", friction=0.0))
        model = builder.finalize()
        solver = _friction_solver(model)
        states, control = [model.state(), model.state()], model.control()
        _advance_friction(solver, states, control, 1)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(states[0], states[1], control, None, DT_JOINT_FRICTION)
        solver.check_constraint_capacity()
        # DOF 0 is world 0's hinge, DOFs 1-6 world 1's free joint and DOF 7 world 2's hinge.
        for dof, value, world in ((0, -1.0, 0), (7, float("nan"), 2), (1, 0.5, 1), (4, float("inf"), 1)):
            with self.subTest(dof=dof, value=value):
                friction = np.zeros(model.joint_dof_count, dtype=np.float32)
                friction[dof] = value
                model.joint_friction.assign(friction)
                wp.capture_launch(capture.graph)
                message = rf"joint_friction was invalid at a step in worlds \[{world}\]"
                with self.assertRaisesRegex(RuntimeError, message):
                    solver.check_constraint_capacity()
                # A reset clears the flag, and the next step flags the unrepaired coefficient again.
                solver.reset(states[0])
                solver.check_constraint_capacity()
                wp.capture_launch(capture.graph)
                with self.assertRaisesRegex(RuntimeError, message):
                    solver.check_constraint_capacity()
                model.joint_friction.zero_()
                wp.capture_launch(capture.graph)
                other = wp.array(np.arange(4) == (world + 1) % 3, dtype=wp.bool, device=model.device)
                solver.reset(states[0], world_mask=other)
                with self.assertRaisesRegex(RuntimeError, message):
                    solver.check_constraint_capacity()
                solver.reset(states[0])
                solver.check_constraint_capacity()
        with self.assertRaises(AttributeError):
            solver.enable_joint_friction = False

    def test_invalid_friction_flags_the_owning_world(self):
        """Attribute loop-closing and global DOFs to their entry of the status layout."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for link_count in (3, 0, 2):
            builder.begin_world()
            if link_count == 0:
                builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
            else:
                links = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(link_count)]
                joints = [builder.add_joint_revolute(-1, link, axis=newton.Axis.Y) for link in links]
                builder.add_articulation(joints)
                loop = builder.add_joint_ball(links[0], links[-1])
            builder.end_world()
        global_link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_revolute(-1, global_link, axis=newton.Axis.Y)])
        model = builder.finalize()
        joint_qd_start = model.joint_qd_start.numpy()
        dofs = {name: int(joint_qd_start[joint]) for name, joint in (("loop", loop), ("chain", joints[1]))}
        dofs["global"] = int(joint_qd_start[model.joint_count - 1])
        solver = _friction_solver(model)
        states, control = [model.state(), model.state()], model.control()
        for name, value, world in (("loop", 0.5, 2), ("chain", -1.0, 2), ("global", -1.0, -1)):
            with self.subTest(dof=name):
                friction = np.zeros(model.joint_dof_count, dtype=np.float32)
                friction[dofs[name]] = value
                model.joint_friction.assign(friction)
                _advance_friction(solver, states, control, 1)
                with self.assertRaisesRegex(RuntimeError, rf"worlds \[{world}\]"):
                    solver.check_constraint_capacity()
                model.joint_friction.zero_()
                solver.reset(states[0])


LOAD_COEFFICIENT = 0.1


INERTIA_LOAD = 0.01


INVALID_LOAD_FRICTION = (
    r"Model\.joint_friction or steel:joint_friction_load_coefficient was invalid at a step in worlds \[-1\]"
)


ARM_LENGTH = 0.3


GRAVITY = 9.81


@unittest.skipUnless(wp.is_cuda_available(), "matrix-free joint friction requires CUDA")
class TestSteelJointFrictionLoad(unittest.TestCase):
    def test_sticks_and_slides_by_transmitted_load(self):
        """Hold while tau <= c (m g + tau) and slide against that bound, as PhysX's coefficient law does."""
        mass = 1.0
        for ratio in (0.5, 1.0, 1.5, 4.0):
            with self.subTest(ratio=ratio):
                effort = ratio * LOAD_COEFFICIENT * mass * GRAVITY
                friction = _hinge_friction(_lateral_hinge_model(mass), effort)
                expected = min(effort, LOAD_COEFFICIENT * (mass * GRAVITY + effort))
                self.assertAlmostEqual(friction, expected, delta=1.0e-3 * expected)

    def test_load_is_taken_about_the_child_com(self):
        """Leave out the moment of the reaction about the joint anchor: a COM offset along the axis changes nothing."""
        effort = 2.0 * LOAD_COEFFICIENT * GRAVITY
        centered = _hinge_friction(_lateral_hinge_model(1.0), effort)
        offset = _hinge_friction(_lateral_hinge_model(1.0, axial_offset=0.3), effort)
        self.assertAlmostEqual(offset, centered, delta=1.0e-4)

    def test_joint_effort_loads_its_own_friction(self):
        """Without gravity, the bound is the coefficient times the joint's own applied effort."""
        model = _lateral_hinge_model(1.0, gravity=0.0)
        for effort in (0.5, 2.0):
            with self.subTest(effort=effort):
                self.assertAlmostEqual(_hinge_friction(model, effort), LOAD_COEFFICIENT * effort, delta=1.0e-4)

    def test_load_includes_the_child_subtree(self):
        """Carry the weight and moment of bodies welded below the child."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
        SolverSteel.register_custom_attributes(builder)
        link = builder.add_link(mass=0.4, inertia=wp.mat33(np.eye(3) * INERTIA_LOAD))
        tip = builder.add_link(mass=0.6, inertia=wp.mat33(np.eye(3) * INERTIA_LOAD))
        hinge = builder.add_joint_revolute(
            -1, link, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity())
        )
        weld = builder.add_joint_fixed(link, tip, parent_xform=wp.transform((0.0, 0.2, 0.0), wp.quat_identity()))
        builder.add_articulation([hinge, weld])
        model = builder.finalize()
        _load_coefficient(model).fill_(LOAD_COEFFICIENT)
        solver = _friction_solver(model)
        _advance_hinge(solver, model, 0.0, 1)
        # The tip's weight adds to F and, through its 0.2 m offset, a moment about the link COM to T.
        expected = LOAD_COEFFICIENT * ((0.4 + 0.6) * GRAVITY + 0.6 * GRAVITY * 0.2)
        self.assertAlmostEqual(float(solver._joint_friction_total.numpy()[0]), expected, delta=1.0e-4)

    def test_ball_axes_use_the_shared_load(self):
        """Give each BALL axis the bound of the joint's load: stick below it and slide above it."""
        coefficient = LOAD_COEFFICIENT
        weight = GRAVITY
        for scale, sticks in ((0.5, True), (3.0, False)):
            with self.subTest(scale=scale):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
                SolverSteel.register_custom_attributes(builder)
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA_LOAD))
                builder.add_articulation([builder.add_joint_ball(-1, body)])
                model = builder.finalize()
                _load_coefficient(model).fill_(coefficient)
                solver = _friction_solver(model)
                effort = scale * coefficient * weight
                _, qd = _advance_hinge(solver, model, effort, 5, dofs=3)
                # The bound c (m g + |tau|) holds tau = 0.5 c m g per axis and lets 3 c m g slide.
                if sticks:
                    np.testing.assert_allclose(qd, 0.0, atol=1.0e-6)
                else:
                    bound = coefficient * (weight + np.sqrt(3.0) * effort)
                    expected = (effort - bound) / INERTIA_LOAD * DT_JOINT_FRICTION * 5
                    np.testing.assert_allclose(qd[-1], expected, rtol=1.0e-3)

    def test_moving_joint_with_a_weak_drive_is_stopped(self):
        """Decelerate a spinning hinge by c m g / I in one step, however weak its drive, as PhysX does."""
        for ke in (0.0, 1.0e-6):
            with self.subTest(ke=ke):
                model = _lateral_hinge_model(1.0, ke=ke)
                solver = _friction_solver(model, drive_mode="rows")
                states, control = [model.state(), model.state()], model.control()
                states[0].joint_qd.fill_(1.0)
                _, qd = _advance_friction(solver, states, control, 2)
                expected = 1.0 - LOAD_COEFFICIENT * GRAVITY / INERTIA_LOAD * DT_JOINT_FRICTION
                self.assertAlmostEqual(float(qd[0]), expected, delta=1.0e-4)
                self.assertAlmostEqual(float(qd[1]), 0.0, delta=1.0e-4)

    def test_zero_coefficients_match_unregistered_bitwise(self):
        """Keep a model with registered zero coefficients bit-identical to one without the attribute."""
        runs = []
        for registered in (False, True):
            model = _payload_hinge_model(2.0, coefficient=0.0 if registered else None, ground=True, boxes=3)
            solver = _friction_solver(model, deterministic=wp.DeterministicMode.RUN_TO_RUN)
            self.assertFalse(solver._joint_friction_load_active)
            observables = solver.observables({SolverBase.ObservableKind.BODY_PARENT_F})
            pipeline = newton.CollisionPipeline(model, deterministic=True, rigid_contact_max=64)
            runs.append(
                (*_advance_hinge(solver, model, 0.2, 30, observables, pipeline), observables.body_parent_f.numpy())
            )
        for unregistered, registered in zip(*runs, strict=True):
            np.testing.assert_array_equal(unregistered, registered)

    def test_deterministic_runs_are_bitwise_identical(self):
        """Reproduce a contact-loaded hinge bit for bit under deterministic=True."""
        runs = []
        for _ in range(2):
            model = _payload_hinge_model(2.0, ground=True, boxes=6)
            solver = _friction_solver(
                model, deterministic=wp.DeterministicMode.RUN_TO_RUN, articulation_constraint_max=64
            )
            pipeline = newton.CollisionPipeline(model, deterministic=True, rigid_contact_max=256)
            q, qd = _advance_hinge(solver, model, 0.5, 40, None, pipeline)
            runs.append((q, qd, solver._joint_friction_total.numpy()))
        self.assertGreater(float(runs[0][2][0]), 0.0)
        for first, second in zip(*runs, strict=True):
            np.testing.assert_array_equal(first, second)

    def test_graph_replay_matches_eager(self):
        """Replay captured steps bit-identically to eager stepping, also after an in-place coefficient edit."""
        model = _lateral_hinge_model(1.0)
        effort = 2.0 * LOAD_COEFFICIENT * GRAVITY
        flags = newton.ModelFlags.JOINT_DOF_PROPERTIES
        solver = _friction_solver(model)
        states, control = [model.state(), model.state()], model.control()
        control.joint_f.fill_(effort)
        _, eager = _advance_friction(solver, states, control, 10)
        # The larger coefficient makes the sliding hinge stick, so a replay that missed the edit diverges.
        _load_coefficient(model).fill_(4.0 * LOAD_COEFFICIENT)
        solver.notify_model_changed(flags)
        _, eager_edited = _advance_friction(solver, states, control, 2)
        _load_coefficient(model).fill_(LOAD_COEFFICIENT)
        solver = _friction_solver(model)
        states, control = [model.state(), model.state()], model.control()
        control.joint_f.fill_(effort)
        _advance_friction(solver, states, control, 2)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(states[0], states[1], control, None, DT_JOINT_FRICTION)
            solver.step(states[1], states[0], control, None, DT_JOINT_FRICTION)
        replayed = []
        for _ in range(4):
            wp.capture_launch(capture.graph)
            replayed.append(states[0].joint_qd.numpy()[0])
        _load_coefficient(model).fill_(4.0 * LOAD_COEFFICIENT)
        solver.notify_model_changed(flags)
        wp.capture_launch(capture.graph)
        replayed.append(states[0].joint_qd.numpy()[0])
        self.assertLess(abs(float(eager_edited[1])), abs(float(eager[-1])))
        np.testing.assert_array_equal(np.array(replayed, dtype=np.float32), [*eager[3::2], eager_edited[1]])

    def test_coefficient_edits_apply_at_the_next_step(self):
        """Use the coefficient current at the step."""
        model = _lateral_hinge_model(1.0)
        solver = _friction_solver(model)
        states, control = [model.state(), model.state()], model.control()
        effort = 0.5 * LOAD_COEFFICIENT * GRAVITY
        control.joint_f.fill_(effort)
        _, qd = _advance_friction(solver, states, control, 3)
        np.testing.assert_allclose(qd, 0.0, atol=1.0e-6)
        _load_coefficient(model).zero_()
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _, qd = _advance_friction(solver, states, control, 1)
        self.assertAlmostEqual(float(qd), effort / INERTIA_LOAD * DT_JOINT_FRICTION, delta=1.0e-5)

    def test_rejects_invalid_coefficients(self):
        """Raise for invalid coefficients, unsupported joints and enabling the mode after construction."""
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and non-negative"):
                _friction_solver(_lateral_hinge_model(1.0, coefficient=value))
        model = _lateral_hinge_model(1.0, coefficient=0.0)
        solver = _friction_solver(model)
        _load_coefficient(model).fill_(LOAD_COEFFICIENT)
        with self.assertRaisesRegex(ValueError, "zero at construction"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        builder = newton.ModelBuilder()
        SolverSteel.register_custom_attributes(builder)
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA))
        builder.add_articulation([builder.add_joint_free(body)])
        free = builder.finalize()
        _load_coefficient(free).fill_(LOAD_COEFFICIENT)
        with self.assertRaisesRegex(
            ValueError, r"steel:joint_friction_load_coefficient is nonzero on joint 0 \(FREE\)"
        ):
            _friction_solver(free)

    def test_invalid_coefficient_edits_flag_the_world(self):
        """Flag a step that reads an invalid coefficient, also when the mode was off at construction."""
        for coefficient, edit in (
            (LOAD_COEFFICIENT, float("nan")),
            (LOAD_COEFFICIENT, float("inf")),
            (0.0, LOAD_COEFFICIENT),
        ):
            with self.subTest(coefficient=coefficient, edit=edit):
                model = _lateral_hinge_model(1.0, coefficient=coefficient)
                solver = _friction_solver(model)
                states, control = [model.state(), model.state()], model.control()
                _advance_friction(solver, states, control, 1)
                solver.check_constraint_capacity()
                _load_coefficient(model).fill_(edit)
                _advance_friction(solver, states, control, 1)
                with self.assertRaisesRegex(RuntimeError, INVALID_LOAD_FRICTION):
                    solver.check_constraint_capacity()

    def test_overflowing_load_term_keeps_the_row(self):
        """Saturate a finite coefficient whose load term overflows float32 instead of dropping its row."""
        model = _lateral_hinge_model(1.0, coefficient=1.0e38)
        solver = _friction_solver(model)
        _, qd = _advance_hinge(solver, model, 1.0, 3)
        np.testing.assert_allclose(qd, 0.0, atol=1.0e-6)
        self.assertTrue(np.isfinite(solver._joint_friction_total.numpy()[0]))
        solver.check_constraint_capacity()

    def test_invalid_coefficient_leaves_the_constant_friction(self):
        """Keep the constant friction row of a DOF whose load coefficient was edited to an invalid value."""
        builder = _lateral_hinge_builder(1.0)
        builder.joint_friction = [0.5 * GRAVITY * LOAD_COEFFICIENT]
        model = builder.finalize()
        _load_coefficient(model).fill_(LOAD_COEFFICIENT)
        solver = _friction_solver(model)
        effort = 0.3 * LOAD_COEFFICIENT * GRAVITY
        for value in (float("inf"), float("nan"), -1.0):
            with self.subTest(value=value):
                _load_coefficient(model).fill_(value)
                _, qd = _advance_hinge(solver, model, effort, 3)
                np.testing.assert_allclose(qd, 0.0, atol=1.0e-6)

    def test_invalid_friction_keeps_no_row_with_a_load_coefficient(self):
        """Build no row for an infinite joint_friction even when a load coefficient applies."""
        model = _lateral_hinge_model(1.0)
        solver = _friction_solver(model)
        model.joint_friction.fill_(float("inf"))
        effort = 0.5 * LOAD_COEFFICIENT * GRAVITY
        _, qd = _advance_hinge(solver, model, effort, 1)
        self.assertAlmostEqual(float(qd), effort / INERTIA_LOAD * DT_JOINT_FRICTION, delta=1.0e-5)
        self.assertEqual(float(solver._joint_friction_total.numpy()[0]), float("inf"))
        with self.assertRaisesRegex(RuntimeError, INVALID_LOAD_FRICTION):
            solver.check_constraint_capacity()

    def test_rejects_wrong_shape_replacement(self):
        """Raise before any launch when a coefficient array is replaced with a wrong shape."""
        flags = newton.ModelFlags.JOINT_DOF_PROPERTIES
        for name, count, at_step in (
            ("joint_friction", 0, False),
            ("joint_friction", 2, True),
            ("joint_friction_load_coefficient", 2, False),
            ("joint_friction_load_coefficient", 0, True),
        ):
            label = name if name == "joint_friction" else f"steel:{name}"
            with self.subTest(name=name, count=count, at_step=at_step):
                model = _lateral_hinge_model(1.0)
                solver = _friction_solver(model)
                owner = model if name == "joint_friction" else model.steel
                setattr(owner, name, wp.full(count, 0.5, dtype=wp.float32, device=model.device))
                with self.assertRaisesRegex(ValueError, rf"{label} has shape \({count},\), expected \(1,\)"):
                    if at_step:
                        states, control = [model.state(), model.state()], model.control()
                        _advance_friction(solver, states, control, 1)
                    else:
                        solver.notify_model_changed(flags)

    def test_unregistered_coefficient_notifies_and_steps(self):
        """Notify eagerly and under capture, and step, without the coefficient attribute."""
        model = _payload_hinge_model(2.0, coefficient=None)
        solver = _friction_solver(model)
        flags = newton.ModelFlags.JOINT_DOF_PROPERTIES
        solver.notify_model_changed(flags)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.notify_model_changed(flags)
        wp.capture_launch(capture.graph)
        _advance_hinge(solver, model, 0.2, 2)
        solver.check_constraint_capacity()

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_reads_only_the_steel_coefficient(self):
        """Import the authored Steel coefficient and leave physxJoint:jointFriction unmapped."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        authored = (
            ("steel", "newton:steel:jointFrictionLoadCoefficient", 0.3),
            ("physx", "physxJoint:jointFriction", 0.5),
        )
        for i, (name, attribute, value) in enumerate(authored):
            body = UsdGeom.Cube.Define(stage, f"/World/{name}")
            body.AddTranslateOp().Set((2.0 * i, 0.0, 1.0))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
            mass.CreateMassAttr(1.0)
            mass.CreateDiagonalInertiaAttr((0.1, 0.1, 0.1))
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/{name}_hinge")
            joint.CreateBody1Rel().SetTargets([body.GetPath()])
            joint.CreateAxisAttr("Y")
            joint.CreateLocalPos0Attr((2.0 * i, 0.0, 1.0))
            joint.GetPrim().CreateAttribute(attribute, Sdf.ValueTypeNames.Float).Set(value)
        builder = newton.ModelBuilder()
        SolverSteel.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()
        coefficients = dict(zip(model.joint_label, _load_coefficient(model).numpy().tolist(), strict=True))
        self.assertAlmostEqual(coefficients["/World/steel_hinge"], 0.3, places=6)
        self.assertEqual(coefficients["/World/physx_hinge"], 0.0)

    def test_disabled_friction_ignores_coefficients(self):
        """Leave the coefficients unused when enable_joint_friction is off."""
        model = _lateral_hinge_model(1.0)
        effort = 0.5 * LOAD_COEFFICIENT * GRAVITY
        _, qd = _advance_hinge(_friction_solver(model, enable_joint_friction=False), model, effort, 5)
        np.testing.assert_allclose(qd, effort / INERTIA_LOAD * DT_JOINT_FRICTION * np.arange(1, 6), rtol=1.0e-4)


def _friction_solver(model, **overrides):
    options = {"enable_joint_friction": True, "iterations": 16}
    options.update(overrides)
    solver = SolverSteel(model, **options)
    solver._rigid_body_angular_damping.zero_()
    return solver


def _advance_friction(solver, states, control, steps, dt=DT_JOINT_FRICTION):
    q, qd = [], []
    for _ in range(steps):
        states[0].clear_forces()
        solver.step(states[0], states[1], control, None, dt)
        states.reverse()
        q.append(states[0].joint_q.numpy().copy())
        qd.append(states[0].joint_qd.numpy().copy())
    return np.squeeze(np.array(q)), np.squeeze(np.array(qd))


def _run_friction(model, *, effort=0.0, qd0=0.0, steps, dt=DT_JOINT_FRICTION, dofs=1, **overrides):
    solver = _friction_solver(model, **overrides)
    states, control = [model.state(), model.state()], model.control()
    states[0].joint_qd.assign(np.broadcast_to(np.asarray(qd0, dtype=np.float32), (dofs,)).copy())
    control.joint_f.assign(np.broadcast_to(np.asarray(effort, dtype=np.float32), (dofs,)).copy())
    return _advance_friction(solver, states, control, steps, dt)[1]


def _single_dof_builder(joint, *, friction=FRICTION, ke=0.0, kd=0.0, limit=None, velocity_limit=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA))
    kwargs = {"friction": friction, "armature": 0.0, "target_ke": ke, "target_kd": kd}
    if limit is not None:
        kwargs.update(limit_lower=limit[0], limit_upper=limit[1])
    if velocity_limit is not None:
        kwargs["velocity_limit"] = velocity_limit
    if joint == "revolute":
        index = builder.add_joint_revolute(-1, body, axis=(0.0, 0.0, 1.0), **kwargs)
    else:
        index = builder.add_joint_prismatic(-1, body, axis=(1.0, 0.0, 0.0), **kwargs)
    builder.add_articulation([index])
    return builder


def _single_dof_model(joint, **kwargs):
    return _single_dof_builder(joint, **kwargs).finalize()


def _worlds_model(joint, world_count, **kwargs):
    """Replicate the single-DOF fixture so independent cases share one solver."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for _ in range(world_count):
        builder.add_world(_single_dof_builder(joint, **kwargs))
    return builder.finalize()


def _two_link_builder(*, length, mass, friction, ke=0.0, kd=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3) * INERTIA)
    first = builder.add_link(mass=mass, inertia=inertia)
    second = builder.add_link(mass=mass, inertia=inertia)
    common = {"axis": (0.0, 0.0, 1.0), "armature": 0.0, "target_ke": ke, "target_kd": kd}
    j0 = builder.add_joint_revolute(-1, first, friction=friction[0], limit_lower=-2.0, limit_upper=2.0, **common)
    j1 = builder.add_joint_revolute(
        first,
        second,
        parent_xform=wp.transform((length, 0.0, 0.0), wp.quat_identity()),
        friction=friction[1],
        limit_lower=-2.0,
        limit_upper=2.0,
        **common,
    )
    builder.add_articulation([j0, j1])
    return builder


def _two_link_model(**kwargs):
    return _two_link_builder(**kwargs).finalize()


def _vertical_slider_model(*, mass, half, friction):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_link(mass=mass, inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        friction=friction,
        armature=0.0,
    )
    builder.add_articulation([joint])
    return builder.finalize()


def _pendulum_model(*, friction, mass=1.0, offset=0.3, inertia=0.01, fixed_base=False):
    """Hinge about +y with the center of mass on the link's +x axis, optionally below a fixed base link."""
    builder = newton.ModelBuilder()
    pivot = wp.transform((0.0, 0.0, 1.0), wp.quat_identity())
    joints = []
    parent = -1
    if fixed_base:
        # A fixed root anchors the island so it may sleep.
        parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
        joints.append(builder.add_joint_fixed(-1, parent, parent_xform=pivot))
        pivot = wp.transform_identity()
    body = builder.add_link(mass=mass, com=wp.vec3(offset, 0.0, 0.0), inertia=wp.mat33(np.eye(3) * inertia))
    joints.append(
        builder.add_joint_revolute(
            parent, body, parent_xform=pivot, axis=(0.0, 1.0, 0.0), friction=friction, armature=0.0
        )
    )
    builder.add_articulation(joints)
    return builder.finalize()


def _root_joint_builder(joint, *, friction):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA))
    index = builder.add_joint_ball(-1, body) if joint == "ball" else builder.add_joint_free(body)
    builder.add_articulation([index])
    builder.joint_friction = [friction] * len(builder.joint_friction)
    return builder


def _root_joint_model(joint, *, friction):
    return _root_joint_builder(joint, friction=friction).finalize()


def _lateral_hinge_builder(mass, *, axial_offset=0.0, inertia=INERTIA_LOAD, gravity=GRAVITY, ke=0.0, kd=0.0):
    """Horizontal hinge about y with its COM on the axis, so gravity loads it without a torque about the axis."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    SolverSteel.register_custom_attributes(builder)
    body = builder.add_link(mass=mass, com=wp.vec3(0.0, axial_offset, 0.0), inertia=wp.mat33(np.eye(3) * inertia))
    joint = builder.add_joint_revolute(
        -1,
        body,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        friction=0.0,
        armature=0.0,
        target_ke=ke,
        target_kd=kd,
        effort_limit=1.0 if ke or kd else 1.0e6,
    )
    builder.add_articulation([joint])
    return builder


def _lateral_hinge_model(mass, *, coefficient=LOAD_COEFFICIENT, **kwargs):
    model = _lateral_hinge_builder(mass, **kwargs).finalize()
    _load_coefficient(model).fill_(coefficient)
    return model


def _hinge_friction(model, effort, steps=20):
    """Return tau - I * alpha of a single hinge driven by a constant effort, from its late-step acceleration."""
    _, qd = _advance_hinge(_friction_solver(model), model, effort, steps)
    axis_inertia = float(model.body_inertia.numpy()[0][1][1])
    return effort - axis_inertia * float(qd[-1] - qd[-11]) / (10 * DT_JOINT_FRICTION)


def _payload_hinge_builder(mass, *, ground=False, boxes=0, register=True):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    if register:
        SolverSteel.register_custom_attributes(builder)
    body = builder.add_link(mass=mass, com=wp.vec3(ARM_LENGTH, 0.0, 0.0), inertia=wp.mat33(np.eye(3) * 0.01))
    joint = builder.add_joint_revolute(
        -1,
        body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        friction=0.0,
        armature=0.0,
    )
    builder.add_articulation([joint])
    if ground:
        # A massless plate under the payload carries the boxes resting on it.
        plate = newton.ModelBuilder.ShapeConfig(density=0.0)
        builder.add_shape_box(
            body, xform=wp.transform((ARM_LENGTH, 0.0, -0.02), wp.quat_identity()), hx=0.15, hy=0.15, hz=0.02, cfg=plate
        )
        for i in range(boxes):
            box = builder.add_body(
                xform=wp.transform((ARM_LENGTH + 0.08 * (i % 3 - 1), 0.08 * (i // 3) - 0.04, 0.53), wp.quat_identity())
            )
            builder.add_shape_box(box, hx=0.03, hy=0.03, hz=0.03)
    return builder


def _payload_hinge_model(mass, *, coefficient=LOAD_COEFFICIENT, ground=False, boxes=0):
    if coefficient is None:
        return _payload_hinge_builder(mass, ground=ground, boxes=boxes, register=False).finalize()
    model = _payload_hinge_builder(mass, ground=ground, boxes=boxes).finalize()
    _load_coefficient(model)[:1].fill_(coefficient)
    return model


def _load_coefficient(model):
    return model.steel.joint_friction_load_coefficient


def _advance_hinge(solver, model, effort, steps, observables=None, pipeline=None, dofs=1):
    states, control = [model.state(), model.state()], model.control()
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control.joint_f.fill_(effort)
    contacts = None if pipeline is None else pipeline.contacts()
    q, qd = [], []
    for _ in range(steps):
        states[0].clear_forces()
        if pipeline is not None:
            pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT_JOINT_FRICTION, observables=observables)
        states.reverse()
        q.append(states[0].joint_q.numpy().copy())
        qd.append(states[0].joint_qd.numpy().copy())
    return np.squeeze(np.array(q)[:, :dofs]), np.squeeze(np.array(qd)[:, :dofs])


def _arm_and_box_model():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.2, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1, link, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), friction=0.1
    )
    builder.add_articulation([joint])
    box = builder.add_body(xform=wp.transform((0.6, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize()


if __name__ == "__main__":
    unittest.main()
