# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dt-aware matrix-free warm-start carry."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    gather_mf_warmstart,
)
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def _stack(device, restitution=0.0, drop=False):
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.003
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7, restitution=restitution)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, restitution=restitution))
    for k in range(3):
        z = 0.0505 + 0.101 * k + (0.15 if drop and k == 2 else 0.0)
        b = builder.add_body(xform=wp.transform(wp.vec3(0.002 * k, 0.0, z), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=128,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=8, pgs_velocity_iterations=2, mf_warmstart=True
    )
    return model, pipeline, solver


def _normal_impulse_total(solver):
    imp = solver.mf_impulses.numpy()[0]
    rt = solver.mf_row_type.numpy()[0]
    return float(imp[rt == PGS_CONSTRAINT_TYPE_CONTACT].sum())


def test_warm_carry_kernel_scales_rows_exactly(test: unittest.TestCase, device):
    """The gather kernel scales matched normal and friction rows before any
    solver iteration can conceal a missing or incorrect carry scale."""
    with wp.ScopedDevice(device):
        max_c = 8
        prev_impulses = np.zeros((1, max_c), dtype=np.float32)
        prev_impulses[0, 4:7] = (1.25, -0.5, 0.75)

        prev_types = np.full((1, max_c), -1, dtype=np.int32)
        prev_types[0, 4] = PGS_CONSTRAINT_TYPE_CONTACT
        prev_types[0, 5:7] = PGS_CONSTRAINT_TYPE_FRICTION
        prev_parents = np.full((1, max_c), -1, dtype=np.int32)
        prev_parents[0, 5:7] = 4

        current_types = np.full((1, max_c), -1, dtype=np.int32)
        current_types[0, 2] = PGS_CONSTRAINT_TYPE_CONTACT
        current_types[0, 3:5] = PGS_CONSTRAINT_TYPE_FRICTION
        current_parents = np.full((1, max_c), -1, dtype=np.int32)
        current_parents[0, 3:5] = 2

        impulses = wp.zeros((1, max_c), dtype=wp.float32)
        wp.launch(
            gather_mf_warmstart,
            dim=1,
            inputs=[
                wp.array([1], dtype=wp.int32),  # contact_count
                wp.array([1], dtype=wp.int32),  # MF contact_path
                wp.array([2], dtype=wp.int32),  # current base slot
                wp.array([0], dtype=wp.int32),  # contact_world
                wp.array([0], dtype=wp.int32),  # current -> previous contact
                wp.zeros(1, dtype=wp.int32),  # matched against the solved generation 0
                wp.ones(1, dtype=wp.int32),  # contact generation 1
                1,  # contact stream
                wp.zeros(1, dtype=wp.int32),  # solved generation 0
                wp.ones(1, dtype=wp.int32),  # solved stream
                wp.array([4], dtype=wp.int32),  # previous base slot
                wp.array(prev_impulses, dtype=wp.float32),
                wp.array(prev_types, dtype=wp.int32),
                wp.array(prev_parents, dtype=wp.int32),
                wp.array([5], dtype=wp.int32),
                wp.array(current_types, dtype=wp.int32),
                wp.array(current_parents, dtype=wp.int32),
                wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
                wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
                wp.full((1, max_c), 100.0, dtype=wp.float32),
                0.75,
                4.0,
                max_c,
            ],
            outputs=[impulses],
            device=device,
        )

        carried = impulses.numpy()[0]
        np.testing.assert_allclose(carried[2:5], 3.0 * prev_impulses[0, 4:7], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(carried[[0, 1, 5, 6, 7]], np.zeros(5, dtype=np.float32))


def test_warm_carry_fixed_dt_identity(test: unittest.TestCase, device):
    """At fixed dt the carry ratio is exactly 1: a settled stack's converged
    contact impulses are a fixed point (steady step to step), and the stack
    holds its height."""
    with wp.ScopedDevice(device):
        model, pipeline, solver = _stack(device)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        dt = 1.0 / 60.0
        totals = []
        zs = []
        for k in range(120):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
            if k >= 100:
                totals.append(_normal_impulse_total(solver))
                zs.append(float(s0.body_q.numpy()[2][2]))
        # per-step totals carry ~2-3% GS convergence jitter at 8 iterations;
        # a carry-scale defect (any ratio != 1 applied every frame) compounds
        # geometrically and blows far past this band
        test.assertLess(
            (max(totals) - min(totals)) / max(totals), 0.15, f"impulse totals unstable at fixed dt: {totals[-5:]}"
        )
        test.assertAlmostEqual(
            float(np.mean(totals[:10])),
            float(np.mean(totals[10:])),
            delta=0.02 * float(np.mean(totals)),
            msg="impulse mean drifted at fixed dt",
        )
        # steadiness, not absolute height: measured hover noise at this
        # budget is 0.2-0.4 mm and per-step impulse jitter up to ~7%; a
        # carry-scale defect compounds every frame and runs far past both
        test.assertLess(max(zs) - min(zs), 1.0e-3, f"stack height drifting at fixed dt ({zs[-3:]})")


def test_warm_carry_scales_with_dt(test: unittest.TestCase, device):
    """Across a step-size change the carried impulses rescale by exactly
    dt_new/dt_old: a stack settled at dt/4 keeps its height through a switch
    to dt, and the converged contact impulses scale by ~4x (support impulse
    is force x dt)."""
    with wp.ScopedDevice(device):
        model, pipeline, solver = _stack(device)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        dt_small, dt_big = 1.0 / 240.0, 1.0 / 60.0
        for _k in range(240):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt_small)
            s0, s1 = s1, s0
        i_small = _normal_impulse_total(solver)
        z_before = float(s0.body_q.numpy()[2][2])
        zs = []
        for _k in range(60):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt_big)
            s0, s1 = s1, s0
            zs.append(float(s0.body_q.numpy()[2][2]))
        i_big = _normal_impulse_total(solver)
        ratio = i_big / i_small
        test.assertAlmostEqual(ratio, dt_big / dt_small, delta=0.6, msg=f"impulse ratio {ratio:.2f}, want ~4")
        test.assertLess(abs(zs[0] - z_before), 1.0e-3, "stack popped on the timestep switch")
        test.assertLess(abs(zs[-1] - z_before), 2.0e-3, f"stack drifted after the switch (dz={zs[-1] - z_before:.4f})")


def test_warm_carry_no_kick_after_dt_change_on_impact(test: unittest.TestCase, device):
    """Impact impulses are not proportional to dt. Check both timestep
    directions, especially small-to-large where the carry scales a cached
    impact impulse upward, and reject any resulting energy gain."""
    with wp.ScopedDevice(device):
        for dt_before, dt_after in (
            (1.0 / 240.0, 1.0 / 60.0),
            (1.0 / 60.0, 1.0 / 240.0),
        ):
            builder = newton.ModelBuilder()
            builder.rigid_gap = 0.003
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7, restitution=0.8)
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, restitution=0.8))
            drop_h = 0.25
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, drop_h), wp.quat_identity()))
            builder.add_shape_sphere(b, radius=0.05, cfg=cfg)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(
                model,
                reduce_contacts=True,
                rigid_contact_max=32,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="latest",
            )
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(
                model, pgs_mode="matrix_free", pgs_iterations=8, pgs_velocity_iterations=2, mf_warmstart=True
            )
            s0, s1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            dt = dt_before
            impacted = False
            v_in = 0.0
            max_up = 0.0
            max_z = 0.0
            for _k in range(240):
                pipeline.collide(s0, contacts)
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, dt)
                s0, s1 = s1, s0
                vz = float(s0.body_qd.numpy()[b][2])
                if not impacted and vz > 0.0 and v_in < -0.5:
                    impacted = True
                    dt = dt_after
                if not impacted:
                    v_in = min(v_in, vz)
                else:
                    max_up = max(max_up, vz)
                    max_z = max(max_z, float(s0.body_q.numpy()[b][2]))

            transition = f"{dt_before:.6f} -> {dt_after:.6f}"
            test.assertTrue(impacted, f"ball never bounced ({transition})")
            test.assertLess(
                max_up,
                0.8 * abs(v_in) * 1.10,
                f"rebound {max_up:.2f} exceeds e*v_in ({0.8 * abs(v_in):.2f}) after dt {transition}",
            )
            test.assertLess(max_z, drop_h, f"ball gained energy after dt {transition}")


class TestFeatherPGSWarmstartDt(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSWarmstartDt,
    "test_warm_carry_kernel_scales_rows_exactly",
    test_warm_carry_kernel_scales_rows_exactly,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSWarmstartDt,
    "test_warm_carry_fixed_dt_identity",
    test_warm_carry_fixed_dt_identity,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSWarmstartDt,
    "test_warm_carry_scales_with_dt",
    test_warm_carry_scales_with_dt,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSWarmstartDt,
    "test_warm_carry_no_kick_after_dt_change_on_impact",
    test_warm_carry_no_kick_after_dt_change_on_impact,
    devices=get_selected_cuda_test_devices(),
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
