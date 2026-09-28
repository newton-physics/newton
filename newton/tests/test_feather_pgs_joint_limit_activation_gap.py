# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    build_joint_limit_rows_for_size,
)
from newton.solvers import SolverFeatherPGS


def _built_rows(q: float, *, gap: float, lower: float = -1.0, upper: float = 1.0):
    device = "cpu"
    max_constraints = 8
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    J_group = wp.zeros((1, max_constraints, 1), dtype=wp.float32, device=device)
    world_row_type = wp.zeros((1, max_constraints), dtype=wp.int32, device=device)
    world_row_parent = wp.zeros((1, max_constraints), dtype=wp.int32, device=device)
    world_row_mu = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_row_beta = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_row_cfm = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_phi = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_target_velocity = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)

    wp.launch(
        build_joint_limit_rows_for_size,
        dim=1,
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array([lower], dtype=wp.float32, device=device),
            wp.array([upper], dtype=wp.float32, device=device),
            wp.array([q], dtype=wp.float32, device=device),
            gap,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            max_constraints,
            0.2,
            0.0,
        ],
        outputs=[
            world_slot_counter,
            J_group,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_beta,
            world_row_cfm,
            world_phi,
            world_target_velocity,
        ],
        device=device,
    )
    wp.synchronize()
    count = int(world_slot_counter.numpy()[0])
    return J_group.numpy()[0, :count, 0].tolist(), world_phi.numpy()[0, :count].tolist()


def _make_phase_layout_run(
    device,
    pgs_mode,
    *,
    pgs_iterations=0,
    pgs_schedule="interleaved",
    response="immediate",
    enable_joint_velocity_limits=True,
):
    """Build a deterministic scene containing every phase-bounded row family."""
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
        pgs_mode=pgs_mode,
        pgs_schedule=pgs_schedule,
        articulated_contact_response=response,
        enable_joint_limits=True,
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        velocity_limit_activation_fraction=0.5,
        dense_max_constraints=64,
        mf_max_constraints=64,
        pgs_iterations=pgs_iterations,
        pgs_warmstart=False,
        mf_warmstart=False,
    )
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0] = 0.5
    free_joint = int(np.flatnonzero(model.joint_type.numpy() == int(JointType.FREE))[0])
    free_dof = int(model.joint_qd_start.numpy()[free_joint])
    joint_qd[free_dof] = 1.5
    joint_qd[free_dof + 3] = 1.5
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    return model, solver, state_0, state_1, model.control(), pipeline, pipeline.contacts()


def _step_once(run):
    """Build all constraint rows for one deterministic step."""
    _model, solver, state_0, state_1, control, pipeline, contacts = run
    state_0.clear_forces()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
    return solver


def _assert_dense_phase_layout(
    test_case: unittest.TestCase, solver: SolverFeatherPGS, *, expect_joint_velocity_limits: bool = True
):
    """Assert the three dense row-family partitions are ordered and populated."""
    dense_count = int(solver.constraint_count.numpy()[0])
    dense_types = solver.row_type.numpy()[0, :dense_count]
    phase_zero_end, phase_one_end = solver.dense_phase_bounds.numpy()[0]
    limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    contact_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_CONTACT)
    friction_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_FRICTION)

    test_case.assertGreater(limit_rows.size, 0, "expected an active joint position-limit row")
    if expect_joint_velocity_limits:
        test_case.assertGreater(velocity_limit_rows.size, 0, "expected active joint velocity-limit rows")
    else:
        test_case.assertEqual(velocity_limit_rows.size, 0)
    test_case.assertGreater(contact_rows.size, 0, "expected dense contact rows")
    test_case.assertGreater(friction_rows.size, 0, "expected dense friction rows")
    test_case.assertTrue(np.all(limit_rows < phase_zero_end), (limit_rows, phase_zero_end))
    if expect_joint_velocity_limits:
        test_case.assertTrue(
            np.all((phase_zero_end <= velocity_limit_rows) & (velocity_limit_rows < phase_one_end)),
            (velocity_limit_rows, phase_zero_end, phase_one_end),
        )
    test_case.assertTrue(np.all(contact_rows >= phase_one_end), (contact_rows, phase_one_end))
    test_case.assertTrue(np.all(friction_rows >= phase_one_end), (friction_rows, phase_one_end))


def _assert_mf_phase_layout(test_case: unittest.TestCase, solver: SolverFeatherPGS):
    """Assert MF contacts precede the populated rigid velocity-limit tail."""
    mf_count = int(solver.mf_constraint_count.numpy()[0])
    mf_types = solver.mf_row_type.numpy()[0, :mf_count]
    mf_contact_end = int(solver.mf_contact_rows_end.numpy()[0])
    contact_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_CONTACT)
    friction_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_FRICTION)
    velocity_limit_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)

    test_case.assertGreater(contact_rows.size, 0, "expected matrix-free contact rows")
    test_case.assertGreater(friction_rows.size, 0, "expected matrix-free friction rows")
    test_case.assertGreater(velocity_limit_rows.size, 0, "expected active rigid-body velocity-limit rows")
    test_case.assertTrue(np.all(contact_rows < mf_contact_end), (contact_rows, mf_contact_end))
    test_case.assertTrue(np.all(friction_rows < mf_contact_end), (friction_rows, mf_contact_end))
    test_case.assertTrue(np.all(velocity_limit_rows >= mf_contact_end), (velocity_limit_rows, mf_contact_end))
    test_case.assertLess(mf_contact_end, mf_count)


def _run_joint_limit_trajectory(use_warp_builder: bool):
    """Run a short PhysX-grasp propagation trajectory with one row builder."""
    _model, solver, state_in, state_out, control, pipeline, contacts = _make_phase_layout_run(
        "cuda:0",
        "matrix_free",
        pgs_iterations=8,
        pgs_schedule="physx_grasp",
        response="propagation",
    )
    if not use_warp_builder:
        solver._joint_limit_warp_kernels.clear()

    samples = []
    for _ in range(4):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        dense_count = int(solver.constraint_count.numpy()[0])
        mf_count = int(solver.mf_constraint_count.numpy()[0])
        propagation_count = int(solver.propagation_constraint_count.numpy()[0])
        samples.append(
            {
                "counts": (dense_count, mf_count, propagation_count),
                "bounds": solver.dense_phase_bounds.numpy().copy(),
                "row_type": solver.row_type.numpy()[0, :dense_count].copy(),
                "mf_row_type": solver.mf_row_type.numpy()[0, :mf_count].copy(),
                "propagation_row_type": solver.propagation_row_type.numpy()[0, :propagation_count].copy(),
                "impulses": solver.impulses.numpy()[0, :dense_count].copy(),
                "mf_impulses": solver.mf_impulses.numpy()[0, :mf_count].copy(),
                "propagation_impulses": solver.propagation_impulses.numpy()[0, :propagation_count].copy(),
                "joint_q": state_out.joint_q.numpy().copy(),
                "joint_qd": state_out.joint_qd.numpy().copy(),
            }
        )
        state_in, state_out = state_out, state_in
    return solver, samples


class TestFeatherPGSJointLimitActivationGap(unittest.TestCase):
    def test_dense_row_families_respect_phase_bounds(self):
        """Keep active position limits before the CPU dense phase-0 bound."""
        solver = _step_once(_make_phase_layout_run("cpu", "split", enable_joint_velocity_limits=False))
        _assert_dense_phase_layout(self, solver, expect_joint_velocity_limits=False)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free row-family layout requires CUDA")
    def test_combined_row_families_respect_phase_bounds(self):
        """Keep dense and matrix-free row families inside their phase ranges."""
        solver = _step_once(_make_phase_layout_run("cuda:0", "matrix_free"))
        _assert_dense_phase_layout(self, solver)
        _assert_mf_phase_layout(self, solver)

    @unittest.skipUnless(wp.is_cuda_available(), "joint-limit trajectory parity requires CUDA")
    def test_warp_builder_matches_physx_grasp_propagation_trajectory(self):
        """Match scalar joint-limit assembly under PhysX-grasp propagation."""
        reference_solver, reference = _run_joint_limit_trajectory(use_warp_builder=False)
        warp_solver, actual = _run_joint_limit_trajectory(use_warp_builder=True)

        self.assertFalse(reference_solver._joint_limit_warp_kernels)
        self.assertTrue(any(kernel is not None for kernel in warp_solver._joint_limit_warp_kernels.values()))
        self.assertEqual(reference_solver.pgs_schedule, "physx_grasp")
        self.assertEqual(reference_solver.articulated_contact_response, "propagation")
        self.assertEqual(len(reference), len(actual))
        for step, (expected, observed) in enumerate(zip(reference, actual, strict=True)):
            self.assertEqual(observed["counts"], expected["counts"], f"row counts differed at step {step}")
            for label in ("bounds", "row_type", "mf_row_type", "propagation_row_type"):
                np.testing.assert_array_equal(
                    observed[label], expected[label], err_msg=f"{label} differed at step {step}"
                )
            for label in ("impulses", "mf_impulses", "propagation_impulses", "joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    observed[label],
                    expected[label],
                    rtol=1.0e-5,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

        limit_rows = np.flatnonzero(reference[0]["row_type"] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
        self.assertGreater(limit_rows.size, 0)
        self.assertGreater(float(np.max(np.abs(reference[0]["impulses"][limit_rows]))), 0.0)

    def test_solver_rejects_invalid_joint_limit_activation_gap(self):
        model = newton.ModelBuilder().finalize()

        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "joint_limit_activation_gap"):
                    SolverFeatherPGS(model, joint_limit_activation_gap=value)

    def test_finite_gap_builds_only_near_limit_rows(self):
        self.assertEqual(_built_rows(0.0, gap=0.2), ([], []))

        jacobian, phi = _built_rows(-0.85, gap=0.2)
        self.assertEqual(jacobian, [1.0])
        self.assertAlmostEqual(phi[0], 0.15, places=6)

        jacobian, phi = _built_rows(0.85, gap=0.2)
        self.assertEqual(jacobian, [-1.0])
        self.assertAlmostEqual(phi[0], 0.15, places=6)

    def test_finite_gap_does_not_activate_unlimited_sentinel_limits(self):
        self.assertEqual(_built_rows(0.0, gap=0.2, lower=-MAXVAL, upper=MAXVAL), ([], []))

    def test_infinite_gap_preserves_historical_always_allocate_behavior(self):
        jacobian, phi = _built_rows(0.0, gap=float("inf"))

        self.assertEqual(jacobian, [1.0, -1.0])
        self.assertEqual(phi, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
