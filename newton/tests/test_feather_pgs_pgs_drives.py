# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PGS joint drive rows (``drive_mode="physx_pgs"``) of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    prescale_joint_velocity_limits,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _use_resident_mfgs_metadata
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode="physx_pgs", **kwargs)
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


def _build_arm_and_box_model(device) -> newton.Model:
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
    for drive_mode in ("augmented", "physx_pgs"):
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode=drive_mode, pgs_iterations=100)
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, model.control(), None, DT)
        qd[drive_mode] = float(state_1.joint_qd.numpy()[0])
    augmented_mass = MASS + ARMATURE + DT * KD + DT * DT * KE
    test.assertAlmostEqual(qd["augmented"], DT * effort_limit / augmented_mass, delta=1.0e-6)
    test.assertAlmostEqual(qd["physx_pgs"], DT * effort_limit / (MASS + ARMATURE), delta=1.0e-6)
    test.assertGreater(qd["physx_pgs"] - qd["augmented"], 1.0e-3)


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
                solver = SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode="physx_pgs")
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
    solver = SolverFeatherPGS(
        model, pgs_mode="matrix_free", drive_mode="physx_pgs", enable_joint_limits=True, warn_constraint_overflow=False
    )
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    counts = solver.constraint_count.numpy()
    row_type = solver.row_type.numpy()
    np.testing.assert_array_equal(counts, [9, 9])  # 3 drive rows + 2 position-limit rows per DOF
    for world in range(2):
        test.assertTrue(np.all(row_type[world, :3] == PGS_CONSTRAINT_TYPE_JOINT_TARGET))
        test.assertTrue(np.all(row_type[world, 3:9] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))

    with test.assertWarnsRegex(UserWarning, r"need at least 6 dense rows .* dense_max_constraints=2"):
        small = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            drive_mode="physx_pgs",
            enable_joint_limits=True,
            dense_max_constraints=2,
            warn_constraint_overflow=False,
        )
    small.step(model.state(), model.state(), model.control(), None, DT)
    np.testing.assert_array_equal(small.constraint_count.numpy(), [2, 2])
    test.assertTrue(np.all(small.row_type.numpy()[:, :2] == PGS_CONSTRAINT_TYPE_JOINT_TARGET))
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode="physx_pgs", joint_limit_activation_gap=0.0)
    solver.step(model.state(), model.state(), model.control(), None, DT)
    test.assertEqual(int(solver.constraint_count.numpy()[0]), 2)
    np.testing.assert_array_equal(solver.drive_slot.numpy(), [0, -1, 1])


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
        solver = SolverFeatherPGS(
            model, pgs_mode="matrix_free", drive_mode="physx_pgs", enable_joint_limits=True, pgs_cfm=cfm
        )
        solver.step(model.state(), model.state(), model.control(), None, DT)
        diags[cfm] = solver.diag.numpy()[0, : int(solver.constraint_count.numpy()[0])]
    test.assertEqual(diags[0.0].size, 9)  # 3 drive rows + 2 position-limit rows per DOF
    np.testing.assert_array_equal(diags[0.5][:3], diags[0.0][:3])  # drive rows
    np.testing.assert_allclose(diags[0.5][3:] - diags[0.0][3:], 0.5, rtol=1.0e-5)  # limit rows
    # With a large CFM the PGS drive still reaches the backward-Euler reference.
    np.testing.assert_allclose(
        _slider_trajectory(_build_slider(device), 120, pgs_cfm=0.5),
        _backward_euler_trajectory(120),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_fused_clamp_lands_on_the_limit_with_large_cfm(test, device):
    """Return a driven DOF exactly to its limit regardless of pgs_cfm."""
    model = _build_arm_and_box_model(device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        drive_mode="physx_pgs",
        enable_joint_velocity_limits=True,
        pgs_cfm=0.5,
        pgs_iterations=8,
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
    model = _build_arm_and_box_model(device)
    kwargs = {"drive_mode": "physx_pgs", "enable_joint_velocity_limits": True, "pgs_iterations": 8}
    eager = _run_arm_and_box(model, SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs), 20)
    captured = _run_arm_and_box(model, SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs), 20, capture=True)
    np.testing.assert_allclose(captured, eager, atol=2.0e-6)


def test_streamed_drive_metadata_matches_resident(test, device):
    """Produce the same trajectory whether the drive-row metadata is resident or streamed."""
    model = _build_arm_and_box_model(device)
    max_shared = int(getattr(model.device, "max_shared_memory_per_block", 0))
    resident_rows, streamed_rows = 32, 160
    for rows, resident in ((resident_rows, True), (streamed_rows, False)):
        test.assertEqual(
            _use_resident_mfgs_metadata(rows, 32, 7, max_shared, has_drive_rows=True, fuse_vel_limits=True), resident
        )
    kwargs = {"drive_mode": "physx_pgs", "enable_joint_velocity_limits": True, "pgs_iterations": 8}
    reference = _run_arm_and_box(
        model, SolverFeatherPGS(model, pgs_mode="matrix_free", dense_max_constraints=resident_rows, **kwargs), 20
    )
    streamed = _run_arm_and_box(
        model, SolverFeatherPGS(model, pgs_mode="matrix_free", dense_max_constraints=streamed_rows, **kwargs), 20
    )
    np.testing.assert_allclose(streamed, reference, atol=1.0e-6)


class TestFeatherPGSPGSDrives(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
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
    add_function_test(TestFeatherPGSPGSDrives, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main()
