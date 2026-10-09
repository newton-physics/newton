# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Row schedules of the SolverFeatherPGS matrix-free sweep (``pgs_schedule``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

SCHEDULES = ("interleaved", "contact_then_internal", "physx_grasp")
# Scissor link offset whose links start just apart, with speculative contact rows.
_SPECULATIVE_OFFSET = 0.031


def _scene(device, worlds=1, offset=0.028):
    """A limited, driven three-link scissor next to a free body above its speed bound.

    At the default ``offset`` the two links overlap slightly, so the first step has contact rows.
    """
    template = newton.ModelBuilder()
    SolverFeatherPGS.register_custom_attributes(template)
    template.default_shape_cfg.mu = 0.5
    base = template.add_link()
    template.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    joints = [
        template.add_joint_revolute(
            -1,
            base,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            limit_lower=-0.001,
            limit_upper=0.001,
        )
    ]
    for side in (1.0, -1.0):
        link = template.add_link()
        template.add_shape_box(link, hx=0.12, hy=0.03, hz=0.04)
        joints.append(
            template.add_joint_revolute(
                base,
                link,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.06, side * offset, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
                target_ke=50.0 if side > 0 else 0.0,
                target_kd=2.0 if side > 0 else 0.0,
                target_pos=0.3,
            )
        )
    template.add_articulation(joints)
    template.add_body(
        xform=wp.transform(wp.vec3(2.0, 0.0, 2.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 1.0e-2),
        custom_attributes={"rigid_body_max_linear_velocity": 0.001, "rigid_body_max_angular_velocity": 0.001},
    )
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    model = builder.finalize(device=device)
    joint_qd = model.joint_qd.numpy()
    # Fast enough to close the 0.001 rad gap to the upper limit within one step.
    joint_qd[0::4] = 5.0
    model.joint_qd.assign(joint_qd)
    return model


def _solver(model, **kwargs):
    options = {
        "enable_joint_limits": True,
        "friction_anchor_beta": 0.0,
        "drive_mode": "physx_pgs",
        "dense_max_constraints": 64,
    }
    options.update(kwargs)
    return SolverFeatherPGS(model, **options)


def _step(model, solver, steps=1):
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return state_0


def test_schedule_validation(test, device):
    """Accept the three schedules and reject unknown names and unsupported combinations."""
    model = _scene(device)
    for schedule in SCHEDULES:
        test.assertEqual(SolverFeatherPGS(model, pgs_schedule=schedule).pgs_schedule, schedule)
    with test.assertRaisesRegex(ValueError, "pgs_schedule"):
        SolverFeatherPGS(model, pgs_schedule="bad")
    for schedule in SCHEDULES[1:]:
        with test.subTest(schedule=schedule):
            with test.assertRaisesRegex(NotImplementedError, "contact_torsion_radius.*pgs_schedule='interleaved'"):
                SolverFeatherPGS(model, pgs_schedule=schedule, contact_torsion_radius=0.01)
            with test.assertRaisesRegex(NotImplementedError, "contact_compliance.*pgs_schedule='interleaved'"):
                SolverFeatherPGS(model, pgs_schedule=schedule, contact_compliance=True, friction_anchor_beta=0.0)
    with test.assertRaisesRegex(NotImplementedError, "propagation-fused"):
        SolverFeatherPGS(model, pgs_schedule="contact_then_internal", articulated_contact_response="propagation-fused")
    fused = SolverFeatherPGS(model, pgs_schedule="physx_grasp", articulated_contact_response="propagation-fused")
    test.assertEqual(fused.pgs_schedule, "physx_grasp")


def test_split_rejects_schedules(test, device):
    """Reject every schedule but the interleaved one in the split solve."""
    model = _scene(device)
    for schedule in SCHEDULES[1:]:
        with test.subTest(schedule=schedule):
            with test.assertRaisesRegex(NotImplementedError, "pgs_schedule.*requires pgs_mode='matrix_free'"):
                SolverFeatherPGS(model, pgs_mode="split", pgs_schedule=schedule)


def test_row_phases_touch_only_their_row_families(test, device):
    """Restrict each row phase of the matrix-free kernel to its row families.

    The scene has a PGS drive row, an active joint limit, contacts between two links of one
    articulation (dense rows) and a free body above its velocity bound (velocity-limit rows).
    """
    model = _scene(device)
    solver = _solver(model, pgs_iterations=0, pgs_schedule="physx_grasp")
    _step(model, solver)
    dense_count = int(solver.constraint_count.numpy()[0])
    row_type = solver.row_type.numpy()[0, :dense_count]
    families = {
        "drive": row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET,
        "limit": row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
        "contact": (row_type == PGS_CONSTRAINT_TYPE_CONTACT) | (row_type == PGS_CONSTRAINT_TYPE_FRICTION),
    }
    for name, mask in families.items():
        test.assertTrue(mask.any(), name)
    mf_count = int(solver.mf_constraint_count.numpy()[0])
    mf_contact_end = int(solver.mf_contact_rows_end.numpy()[0])
    test.assertGreater(mf_count, mf_contact_end)

    touched = {}
    for row_phase in (1, 2, 3, 4, 5):
        solver.impulses.zero_()
        solver.mf_impulses.zero_()
        wp.copy(solver.v_out, solver.v_hat)
        solver._launch_mf_gs_phase(row_phase)
        dense = np.abs(solver.impulses.numpy()[0, :dense_count]) > 0.0
        velocity_limits = np.abs(solver.mf_impulses.numpy()[0, mf_contact_end:mf_count]) > 0.0
        touched[row_phase] = (*(bool(dense[mask].any()) for mask in families.values()), bool(velocity_limits.any()))
    # (drive, limit, contact, velocity limit)
    test.assertEqual(touched[1], (False, False, True, False))
    test.assertEqual(touched[2], (True, True, False, True))
    test.assertEqual(touched[3], (True, True, False, False))
    test.assertEqual(touched[4], (False, False, True, False))
    test.assertEqual(touched[5], (False, False, False, True))


def test_physx_grasp_keeps_the_interleaved_row_order(test, device):
    """Match the interleaved sweep: the dense layout already puts internal rows before contacts."""
    for iterations in (1, 12):
        results = []
        for schedule in ("interleaved", "physx_grasp"):
            model = _scene(device, worlds=2, offset=_SPECULATIVE_OFFSET)
            results.append(
                _step(model, _solver(model, pgs_iterations=iterations, pgs_schedule=schedule)).joint_qd.numpy()
            )
        np.testing.assert_allclose(results[1], results[0], rtol=0.0, atol=1.0e-6)


def test_contact_then_internal_gives_internal_rows_the_last_word(test, device):
    """Leave the internal rows at their fixed point after the contact iterations, unlike one interleaved sweep."""
    changes = {}
    for schedule, iterations in (("contact_then_internal", 12), ("interleaved", 1)):
        model = _scene(device, offset=_SPECULATIVE_OFFSET)
        solver = _solver(model, pgs_iterations=iterations, pgs_schedule=schedule)
        _step(model, solver)
        dense_count = int(solver.constraint_count.numpy()[0])
        row_type = solver.row_type.numpy()[0, :dense_count]
        internal = (row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET) | (row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
        before = solver.impulses.numpy()[0, :dense_count][internal].copy()
        solver._launch_mf_gs_phase(3)
        after = solver.impulses.numpy()[0, :dense_count][internal]
        changes[schedule] = float(np.abs(after - before).max())
    test.assertLess(changes["contact_then_internal"], 1.0e-4)
    test.assertGreater(changes["interleaved"], 1.0e-3)


def test_propagation_schedules(test, device):
    """Run the propagation response with every schedule; physx_grasp is its interleaved order."""
    results = {}
    for schedule in SCHEDULES:
        model = _scene(device, offset=_SPECULATIVE_OFFSET)
        solver = _solver(
            model,
            drive_mode="augmented",
            pgs_iterations=200,
            pgs_schedule=schedule,
            articulated_contact_response="propagation",
        )
        results[schedule] = _step(model, solver, steps=3).joint_qd.numpy()
        test.assertTrue(np.isfinite(results[schedule]).all())
    np.testing.assert_array_equal(results["physx_grasp"], results["interleaved"])


def test_captured_schedules_match_eager(test, device):
    """Replay every schedule's step in a CUDA graph with the eager result."""
    for schedule in SCHEDULES:
        with test.subTest(schedule=schedule):
            finals = []
            for capture in (False, True):
                model = _scene(device, offset=_SPECULATIVE_OFFSET)
                solver = _solver(model, pgs_iterations=4, pgs_schedule=schedule)
                state_0, state_1 = model.state(), model.state()
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                pipeline = newton.CollisionPipeline(model, deterministic=True)
                contacts = pipeline.contacts()
                control = model.control()

                def substep(
                    state_0=state_0,
                    state_1=state_1,
                    contacts=contacts,
                    solver=solver,
                    pipeline=pipeline,
                    control=control,
                ):
                    pipeline.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
                    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                        wp.copy(getattr(state_0, name), getattr(state_1, name))

                substep()
                if capture:
                    with wp.ScopedCapture(device) as scope:
                        substep()
                    wp.capture_launch(scope.graph)
                else:
                    substep()
                finals.append(state_0.joint_qd.numpy())
            np.testing.assert_allclose(finals[1], finals[0], rtol=0.0, atol=1.0e-5)


class TestFeatherPGSSchedule(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _fn in (
    test_schedule_validation,
    test_row_phases_touch_only_their_row_families,
    test_physx_grasp_keeps_the_interleaved_row_order,
    test_contact_then_internal_gives_internal_rows_the_last_word,
    test_propagation_schedules,
    test_captured_schedules_match_eager,
):
    add_function_test(TestFeatherPGSSchedule, _fn.__name__, _fn, devices=cuda_devices)
add_function_test(
    TestFeatherPGSSchedule, "test_split_rejects_schedules", test_split_rejects_schedules, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main()
