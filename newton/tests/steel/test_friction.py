# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction patches, anchors, directions and friction modes of SolverSteel."""

import itertools
import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.friction_patches import (
    _pose_motion,
    finish_patch_impulses,
    link_patch_rows,
    seed_patch_impulses,
)
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    compute_mf_effective_mass_and_rhs,
    compute_mf_velocity_rhs,
    compute_world_contact_bias,
    compute_world_contact_velocity_bias,
)
from newton._src.solvers.steel.solver_steel import (
    _colored_staging_supported,
    _get_pgs_solve_mf_gs_kernel,
    _get_pgs_solve_propagation_colored_block_kernel,
)
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import (
    _DEVICE,
    _SQUEEZE_SOLVER,
    _build_v_jaws,
    _patch_fixture,
    fixture,
    set_torsion_radius,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _run_squeeze(tilt_deg, steps, dt=0.005, matching="latest", substeps=1, device=None, **solver_kwargs):
    """Return the object's z drift relative to the jaws [m] (positive = up), the solver and the state."""
    model, jaws, box = _build_v_jaws(tilt_deg, device=device)
    kwargs = dict(_SQUEEZE_SOLVER)
    kwargs.update(solver_kwargs)
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn", contact_matching=matching)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    rel = []
    for _ in range(steps):
        pipeline.collide(s0, contacts)
        for _sub in range(substeps):
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt / substeps)
            s0, s1 = s1, s0
        bq = s0.body_q.numpy()
        rel.append(bq[box][2] - bq[jaws[0]][2])
    rel = np.asarray(rel)
    settle = int(0.5 / dt)
    return rel[-1] - rel[settle], solver, s0


def _build_incline(theta_deg: float, mu: float):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.rigid_gap = 0.005
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.radians(theta_deg)))
    # A static slab on the world body is the incline.
    b.add_shape_box(
        -1,
        hx=2.0,
        hy=0.5,
        hz=0.05,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), rot),
        cfg=newton.ModelBuilder.ShapeConfig(mu=mu),
    )
    n = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    box = b.add_body(xform=wp.transform(n * (0.05 + 0.0002), rot), label="box")
    b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu))
    model = b.finalize()
    return model, box, rot


def _run_incline(steps: int, dt: float, **solver_kwargs):
    theta, mu = 30.0, 0.3
    model, box, _rot = _build_incline(theta, mu)
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", iterations=32, baumgarte=0.05, **solver_kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, broad_phase="nxn", contact_matching="latest")
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    v = s0.body_qd.numpy()[box][:3]
    down_slope = np.array([np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))])
    return float(np.dot(v, down_slope)), float(np.linalg.norm(v - np.dot(v, down_slope) * down_slope)), solver, contacts


@unittest.skipUnless(wp.is_cuda_available(), "SolverSteel requires CUDA")
class TestSteelFrictionAnchors(unittest.TestCase):
    def test_explicit_opt_out_keeps_friction_rows_velocity_only(self):
        """Preserve velocity-only rows and omit anchor state with ``friction_anchor_beta=0``."""
        model, _jaws, _box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverSteel(
            model, solve_mode="matrix_free", **_SQUEEZE_SOLVER, friction_anchor_beta=0.0
        )
        self.assertFalse(solver._friction_anchors_enabled)
        self.assertFalse(hasattr(solver._friction_patches, "fixed_normal"))
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn")
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        for _ in range(40):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, 0.005)
            s0, s1 = s1, s0
        self.assertEqual(float(np.abs(solver._friction_patches.view.phi.numpy()).max()), 0.0)
        row_type = solver._row_type.numpy()
        friction = row_type == PGS_CONSTRAINT_TYPE_FRICTION
        self.assertGreater(int(friction.sum()), 0)
        self.assertEqual(float(np.abs(solver._phi.numpy()[friction]).max()), 0.0)

    def test_anchors_do_not_require_contact_matching(self):
        """Keep patch history owned by the solver, independently of collision matching."""
        model, _jaws, _box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverSteel(
            model, solve_mode="matrix_free", **_SQUEEZE_SOLVER, friction_anchor_beta=0.2
        )
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, model.state(), model.control(), contacts, 0.005)

    def test_anchors_stop_tangential_drift_of_a_held_box(self):
        """Bound tangential drift in a tilted pinch while leaving flat-jaw behavior unaffected."""
        steps = int(3.0 / 0.005)
        drift_off, _, _ = _run_squeeze(5.0, steps, friction_anchor_beta=0.0)
        drift_on, solver_on, _ = _run_squeeze(5.0, steps, friction_anchor_beta=0.05)
        drift_on_02, _, _ = _run_squeeze(5.0, steps, friction_anchor_beta=0.2)
        self.assertGreater(
            abs(drift_off), 5.0e-4, f"tilted pinch should leak drift without anchors, got {drift_off:.2e} m"
        )
        self.assertLess(
            abs(drift_on),
            0.03 * abs(drift_off),
            f"anchored pinch (beta 0.05) drifted {drift_on:.2e} m vs {drift_off:.2e} m",
        )
        self.assertLess(abs(drift_on_02), 2.0e-5, f"anchored pinch (beta 0.2) drifted {drift_on_02:.2e} m")
        # The held contacts are anchored (not sliding) at the end of the hold.
        valid = solver_on._friction_patches.current.valid.numpy()
        self.assertGreater(int(valid.sum()), 0)
        # Sticky matching replays body-local witness points; anchors must still hold.
        drift_sticky, _, _ = _run_squeeze(5.0, steps, matching="sticky", friction_anchor_beta=0.2)
        self.assertLess(abs(drift_sticky), 2.0e-5, f"anchored pinch (sticky matching) drifted {drift_sticky:.2e} m")
        # Two solver substeps per collision pass: anchors map one to one on the reused
        # contact buffer instead of through its stale match index.
        drift_sub, _, _ = _run_squeeze(5.0, steps, substeps=2, friction_anchor_beta=0.2)
        self.assertLess(abs(drift_sub), 2.0e-5, f"anchored pinch with 2 substeps per collide drifted {drift_sub:.2e} m")
        flat_off, _, _ = _run_squeeze(0.0, steps, friction_anchor_beta=0.0)
        flat_on, _, _ = _run_squeeze(0.0, steps, friction_anchor_beta=0.05)
        self.assertLess(
            abs(flat_on), 0.03 * abs(drift_off), f"anchored flat pinch drifted {flat_on:.2e} m vs {drift_off:.2e} m"
        )
        self.assertLess(abs(flat_on), 5.0e-5, f"anchored flat pinch drifted {flat_on:.2e} m")
        self.assertLess(abs(flat_off), 0.25 * abs(drift_off))

    def test_anchors_hold_a_pinched_box_at_a_coarse_timestep(self):
        """Hold the tilted pinch at 60 Hz without re-anchoring below the Baumgarte equilibrium separation."""
        dt = 1.0 / 60.0
        drift, solver, _ = _run_squeeze(5.0, int(3.0 / dt), dt=dt, friction_anchor_beta=0.05)
        self.assertLess(abs(drift), 1.0e-3, f"anchored pinch at 60 Hz drifted {drift:.2e} m")
        sources = solver._friction_patches.current.source.numpy()
        self.assertGreater(int((sources >= 0).sum()), 0, "anchors were re-created instead of carried")

    def test_anchors_do_not_oppose_genuine_sliding(self):
        """Release saturated anchors on a 30-degree incline and preserve sliding acceleration."""
        dt, steps = 0.005, 200
        v_off, _lat_off, _, _ = _run_incline(steps, dt, friction_anchor_beta=0.0)
        v_on, lat_on, solver, contacts = _run_incline(steps, dt, friction_anchor_beta=0.2)
        g, th, mu = 9.81, np.radians(30.0), 0.3
        v_ref = g * (np.sin(th) - mu * np.cos(th)) * steps * dt
        self.assertGreater(v_off, 0.5 * v_ref)
        self.assertAlmostEqual(v_on, v_off, delta=0.05 * v_ref)
        self.assertLess(lat_on, 1.0e-3)
        # Every loaded contact slid this step, so its anchor was dropped.
        n = int(contacts.rigid_contact_count.numpy()[0])
        if n > 0:
            self.assertEqual(int(solver._friction_patches.current.valid.numpy()[:n].sum()), 0)

    def test_row_builders_store_anchor_separation_per_route(self):
        """Store raw separation in dense rows and gain-scaled separation in free-body rows."""
        beta = 0.3

        def check(model, solver_kwargs, steps=120):
            solver = newton.solvers.SolverSteel(
                model, solve_mode="matrix_free", friction_anchor_beta=beta, **solver_kwargs
            )
            self.assertEqual(solver.friction_anchor_beta, beta)
            pipeline = newton.CollisionPipeline(
                model, rigid_contact_max=256, broad_phase="nxn", contact_matching="latest"
            )
            contacts = pipeline.contacts()
            s0, s1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            for _ in range(steps):
                pipeline.collide(s0, contacts)
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, 0.005)
                s0, s1 = s1, s0
            n = int(contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(n, 0)
            fa = solver._friction_patches.view.phi.numpy()[:n]
            slot = solver._contact_slot.numpy()[:n]
            path = solver._contact_path.numpy()[:n]
            checked = {0: 0, 1: 0}
            for c in range(n):
                if slot[c] < 0 or path[c] < 0:
                    continue
                if path[c] == 0:
                    rt, ph = solver._row_type.numpy()[0], solver._phi.numpy()[0]
                    if rt[slot[c] + 1] != PGS_CONSTRAINT_TYPE_FRICTION:
                        continue
                    np.testing.assert_allclose(ph[slot[c] + 1 : slot[c] + 3], fa[c], rtol=1.0e-6, atol=1.0e-9)
                else:
                    ph, rt = solver._mf_phi.numpy()[0], solver._mf_row_type.numpy()[0]
                    if rt[slot[c] + 1] != PGS_CONSTRAINT_TYPE_FRICTION:
                        continue
                    np.testing.assert_allclose(ph[slot[c] + 1 : slot[c] + 3], beta * fa[c], rtol=1.0e-5, atol=1.0e-9)
                checked[int(path[c])] += 1
            return checked

        # Dense rows: articulated jaws against a free box.
        model, _, _ = _build_v_jaws(5.0)
        self.assertGreater(check(model, _SQUEEZE_SOLVER)[0], 0)
        # Free-body rows: a free box resting on a static slab.
        model, _, _ = _build_incline(0.0, 0.5)
        self.assertGreater(check(model, {"iterations": 16, "baumgarte": 0.05})[1], 0)

    def test_graph_capture_replays(self):
        """Capture and replay two steps with device-side anchor history under a CUDA graph."""
        model, jaws, box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverSteel(
            model, solve_mode="matrix_free", friction_anchor_beta=0.05, **_SQUEEZE_SOLVER
        )
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

        def step():
            nonlocal s0, s1
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, 0.005)
            s0, s1 = s1, s0

        for _ in range(4):
            step()
        with wp.ScopedCapture(device=model.device) as capture:
            step()
            step()
        for _ in range(200):
            wp.capture_launch(capture.graph)
        bq = s0.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(bq)))
        self.assertLess(abs(bq[box][2] - bq[jaws[0]][2]), 0.01)


def _position_rhs(family: str, *, phi, baumgarte, friction_anchor_beta, dt, device):
    """Position-solve RHS of one world with rows ``[CONTACT, FRICTION, FRICTION]``.

    ``phi`` holds what the row builders store: raw separation for dense rows, and
    gain-premultiplied separation for free-body rows.
    """
    count = wp.array([3], dtype=wp.int32, device=device)
    row_type = wp.array(
        [[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]],
        dtype=wp.int32,
        device=device,
    )
    phi_arr = wp.array([list(phi)], dtype=wp.float32, device=device)

    def zeros3():
        return wp.zeros((1, 3), dtype=wp.float32, device=device)

    def zero_jacobian():
        return wp.zeros((1, 3, 6), dtype=wp.float32, device=device)

    neg1 = wp.full((1, 3), -1, dtype=wp.int32, device=device)
    rhs = zeros3()
    if family == "dense":
        wp.launch(
            compute_world_contact_bias,
            dim=1,
            inputs=[count, phi_arr, row_type, zeros3(), baumgarte, friction_anchor_beta, 1.0, 1.0, dt],
            outputs=[rhs, zeros3()],
            device=device,
        )
    else:
        wp.launch(
            compute_mf_effective_mass_and_rhs,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                zero_jacobian(),
                zero_jacobian(),
                wp.zeros((1,), dtype=wp.spatial_matrix, device=device),
                phi_arr,
                row_type,
                zeros3(),
                zeros3(),
                0,
                wp.array([-1], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.array([float("inf")], dtype=wp.float32, device=device),
                1.0e-6,
                baumgarte,
                1.0,
                dt,
                1.0,
                0.5,
                3,
            ],
            outputs=[zeros3(), zero_jacobian(), zero_jacobian(), rhs, zeros3()],
            device=device,
        )
    return rhs.numpy()[0], count, row_type, phi_arr


def _build_two_world_free_model(device):
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_free(parent=-1, child=body)
    template.add_articulation([joint])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    return builder.finalize(device=device)


@unittest.skipUnless(wp.is_cuda_available(), "SolverSteel requires CUDA")
class TestSteelFrictionAnchorKernels(unittest.TestCase):
    """Kernel-level checks of the anchor bookkeeping."""

    def test_rhs_bias_matches_on_every_row_family_and_vanishes_in_velocity_pass(self):
        """Apply equal positional bias on all row families and drop it in velocity-only passes."""
        device = "cuda:0"
        baumgarte, fa_beta, dt = 0.05, 0.3, 0.005
        phi_n, e0, e1 = -2.0e-3, 1.0e-4, -2.5e-4
        expect_pos = np.array([baumgarte * phi_n / dt, fa_beta * e0 / dt, fa_beta * e1 / dt], dtype=np.float32)
        dense, count, row_type, phi_dense = _position_rhs(
            "dense", phi=[phi_n, e0, e1], baumgarte=baumgarte, friction_anchor_beta=fa_beta, dt=dt, device=device
        )
        np.testing.assert_allclose(dense, expect_pos, rtol=1.0e-5)
        free, _, _, phi_free = _position_rhs(
            "free",
            phi=[phi_n, fa_beta * e0, fa_beta * e1],
            baumgarte=baumgarte,
            friction_anchor_beta=fa_beta,
            dt=dt,
            device=device,
        )
        np.testing.assert_allclose(free, expect_pos, rtol=1.0e-5)

        # The velocity-only iterations drop every position bias, including the anchors'.
        zeros3 = wp.zeros((1, 3), dtype=wp.float32, device=device)
        velocity = wp.zeros((1,), dtype=wp.float32, device=device)
        rhs = wp.zeros((1, 3), dtype=wp.float32, device=device)
        wp.launch(
            compute_world_contact_velocity_bias,
            dim=3,
            inputs=[
                count,
                3,
                wp.zeros((1,), dtype=wp.int32, device=device),
                phi_dense,
                row_type,
                zeros3,
                zeros3,
                velocity,
                velocity,
                wp.zeros((1, 1), dtype=wp.int32, device=device),
                wp.zeros((1, 3, 1), dtype=wp.float32, device=device),
                dt,
                0.5,
            ],
            outputs=[rhs],
            device=device,
        )
        np.testing.assert_allclose(rhs.numpy()[0], 0.0, atol=1.0e-9)
        neg1 = wp.full((1, 3), -1, dtype=wp.int32, device=device)
        rhs.fill_(1.0)
        wp.launch(
            compute_mf_velocity_rhs,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                wp.zeros((1, 3, 6), dtype=wp.float32, device=device),
                wp.zeros((1, 3, 6), dtype=wp.float32, device=device),
                wp.zeros((1, 1), dtype=wp.int32, device=device),
                phi_free,
                row_type,
                zeros3,
                zeros3,
                0,
                dt,
                velocity,
                velocity,
                0.5,
                3,
            ],
            outputs=[rhs],
            device=device,
        )
        np.testing.assert_allclose(rhs.numpy()[0], 0.0, atol=1.0e-9)

    def test_reset_clears_anchor_history_full_and_masked(self):
        """Drop carried anchors of selected worlds on reset, including without warm starting."""
        device = "cuda:0"
        model = _build_two_world_free_model(device)
        solver = newton.solvers.SolverSteel(
            model,
            solve_mode="matrix_free",
            friction_anchor_beta=0.2,
            articulation_constraint_max=4,
            free_body_constraint_max=4,
        )
        self.assertFalse(solver.contact_warmstart)
        previous = solver._friction_patches.previous
        n = previous.valid.shape[0]
        body_world = model.body_world.numpy()
        bodies = np.arange(n, dtype=np.int32) % model.body_count
        worlds = body_world[bodies]
        previous.body_a.assign(bodies)
        previous.body_b.fill_(-1)
        for mask, expect_cleared in (
            (None, (True, True)),
            ((True, False, False), (True, False)),
            ((False, True, False), (False, True)),
            ((False, False, True), (False, False)),
        ):
            previous.valid.fill_(1)
            wm = None if mask is None else wp.array(mask, dtype=wp.bool, device=device)
            solver.reset(model.state(), wm)
            valid = previous.valid.numpy()
            for world, cleared in enumerate(expect_cleared):
                sel = valid[worlds == world]
                np.testing.assert_array_equal(sel, 0 if cleared else 1, err_msg=f"mask={mask} world={world}")


def _slide(device, beta, mesh, angle, articulated=False, *, locked_tangent=False):
    """Slide a symmetric, explicitly massed box under a force above the friction limit.

    Returns the final planar velocity, the push direction and the capacity status.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, gap=0.0, restitution=0.0)
    builder.add_shape_box(-1, hx=2.0, hy=2.0, hz=0.05, xform=wp.transform((0, 0, -0.05)), cfg=cfg)
    inertia = np.diag([(0.2**2 + 0.05**2) / 12] * 2 + [2 * 0.2**2 / 12]).astype(np.float32)
    add_body = builder.add_link if articulated else builder.add_body
    body = add_body(xform=wp.transform((0, 0, 0.025)), mass=1.0, inertia=wp.mat33(inertia), lock_inertia=True)
    if articulated:
        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            parent_xform=wp.transform((0, 0, 0.025)),
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(axis=axis)
                for axis in ((newton.Axis.X, newton.Axis.Z) if locked_tangent else newton.Axis)
            ],
        )
        builder.add_articulation([joint])
    if mesh:
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.1, 0.1, 0.025), cfg=cfg)
    else:
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.025, cfg=cfg)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverSteel(
        model,
        solve_mode="matrix_free",
        friction_anchor_beta=beta,
        iterations=64,
        cfm=0.0 if locked_tangent else 1.0e-6,
        free_body_constraint_max=128,
        articulation_constraint_max=32,
    )
    solver._rigid_body_angular_damping.zero_()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
    contacts = pipeline.contacts()
    state, next_state = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    direction = np.array([np.cos(angle), np.sin(angle)])
    force = np.zeros((1, 6), dtype=np.float32)
    force[0, :2] = direction * 1.2 * 0.3 * 9.81
    hz = 240
    for step in range(hz // 5 + hz // 2):
        state.clear_forces()
        if step >= hz // 5:
            state.body_f.assign(force)
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / hz)
        state, next_state = next_state, state
    return state.body_qd.numpy()[0, :2], direction, solver.constraint_overflow.numpy()


def test_articulated_sliding_direction(test, device):
    """Preserve the sliding direction of an articulated box on the dense rows."""
    velocity, direction, overflow = _slide(device, 0.2, False, np.pi / 4, articulated=True)
    np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
    test.assertFalse(overflow.any())


def test_locked_tangent_preserves_other_friction_direction(test, device):
    """Retain friction along a slider when its first tangent has zero response."""
    for beta in (0.0, 0.2):
        with test.subTest(beta=beta):
            velocity, direction, overflow = _slide(device, beta, False, 0.0, articulated=True, locked_tangent=True)
            np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
            test.assertFalse(overflow.any())


def _solve_contact_block(device, tangent_mass, rhs, initial, capacity=32):
    """Sweep one contact's normal and two tangent rows once through the fused matrix-free kernel.

    The world has three DOFs whose rows are the unit Jacobians ``J = I`` with responses
    ``Y = M`` (``M[0, 0] = 1``, tangent block ``tangent_mass``), so the rows see the
    Delassus matrix ``M``. The starting velocity ``M @ initial`` is consistent with the
    initial impulses, as after a warm start.
    """
    matrix = np.eye(3, dtype=np.float32)
    matrix[1:3, 1:3] = tangent_mass
    J = np.zeros((1, capacity, 3), dtype=np.float32)
    Y = np.zeros((1, capacity, 3), dtype=np.float32)
    J[0, :3] = np.eye(3)
    Y[0, :3] = matrix
    row_type = np.full((1, capacity), -1, dtype=np.int32)
    row_type[0, :3] = [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]
    parents = np.full((1, capacity), -1, dtype=np.int32)
    parents[0, 1:3] = 0
    rhs_rows = np.zeros((1, capacity), dtype=np.float32)
    rhs_rows[0, :3] = rhs
    diag = np.zeros((1, capacity), dtype=np.float32)
    diag[0, :3] = np.diag(matrix)
    impulses_np = np.zeros((1, capacity), dtype=np.float32)
    impulses_np[0, :3] = initial
    impulses = wp.array(impulses_np, dtype=float, device=device)
    v_out = wp.array((matrix @ np.asarray(initial, dtype=np.float32)).astype(np.float32), dtype=float, device=device)
    kernel = _get_pgs_solve_mf_gs_kernel(
        capacity, 1, 3, wp.get_device(device).arch, has_dense_velocity_limit_rows=False, shared_metadata=True
    )
    wp.launch_tiled(
        kernel,
        dim=[1],
        inputs=[
            wp.array([3], dtype=int, device=device),
            wp.array([[0, 1, 2]], dtype=int, device=device),
            wp.array(rhs_rows, dtype=float, device=device),
            wp.array(diag, dtype=float, device=device),
            impulses,
            wp.array(J, dtype=float, device=device),
            wp.array(Y, dtype=float, device=device),
            wp.array(row_type, dtype=int, device=device),
            wp.array(parents, dtype=int, device=device),
            wp.full((1, capacity), 0.3, dtype=float, device=device),
            # Drive-row parameters: no drive rows.
            *(wp.zeros((1, 1), dtype=float, device=device) for _ in range(5)),
            wp.zeros((1,), dtype=int, device=device),
            wp.zeros((1,), dtype=int, device=device),
            wp.zeros((1, 4), dtype=int, device=device),
            wp.zeros((1, 1), dtype=float, device=device),
            wp.zeros((1, 1, 6), dtype=float, device=device),
            wp.zeros((1, 1, 6), dtype=float, device=device),
            wp.zeros((1, 1, 6), dtype=float, device=device),
            wp.zeros((1, 1, 6), dtype=float, device=device),
            wp.zeros((1, 1), dtype=float, device=device),
            wp.ones((1, 1), dtype=float, device=device),
            wp.ones((1, 1), dtype=float, device=device),
            wp.full((1, 1), -1, dtype=int, device=device),
            wp.zeros((1, 1), dtype=float, device=device),
            1,
            1.0,
            0,
            0,
            0,
            0,
            0,
        ],
        outputs=[v_out],
        block_dim=32,
        device=device,
    )
    return impulses.numpy()[0, :3]


def test_unequal_tangent_masses(test, device):
    """Solve sticking and sliding blocks with rotated, unequal tangent masses in one sweep."""
    for angle, small_eigenvalue in (
        (0.0, 1.0),
        (0.37, 1.0),
        (1.1, 1.0),
        (0.37, 0.0),
        (0.0, 0.0),
        (0.0, 1.0e-4),
        (0.0, 1.0e-6),
        (0.0, 1.0e-8),
    ):
        for regime in ("sticking", "sliding", "fixed", "weak_sliding", "sliding_fixed"):
            if regime == "weak_sliding" and small_eigenvalue == 0.0:
                continue
            sliding = regime in ("sliding", "weak_sliding", "sliding_fixed")
            with test.subTest(angle=angle, small_eigenvalue=small_eigenvalue, regime=regime):
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                tangent_mass = (rotation @ np.diag([small_eigenvalue, 7.0]) @ rotation.T).astype(np.float32)
                direction = np.array([0.6, 0.8])
                expected = -(0.3 if sliding else 0.1) * direction
                if regime == "fixed":
                    expected = np.array([0.125, 0.0])
                velocity = 2.0 * direction if sliding else np.zeros(2)
                if regime == "weak_sliding":
                    # A tiny positive KKT multiplier still requires a boundary impulse.
                    velocity = small_eigenvalue * direction
                rhs = np.array([-1.0, *(velocity - tangent_mass @ expected)], dtype=np.float32)
                initial = np.zeros(3, dtype=np.float32)
                if regime in ("fixed", "sliding_fixed"):
                    # The installed velocity M @ initial carries these impulses.
                    initial[:] = [1.0, *expected]
                if regime == "fixed":
                    # Power-of-two impulses make the row residual exactly zero.
                    rhs[1:3] = -(tangent_mass @ expected)
                actual = _solve_contact_block(device, tangent_mass, rhs, initial)
                if regime in ("fixed", "sliding_fixed"):
                    np.testing.assert_allclose(actual, initial, rtol=0.0, atol=1.0e-6)
                elif small_eigenvalue == 0.0 and not sliding:
                    # A singular sticking block has multiple impulse solutions.
                    np.testing.assert_allclose(tangent_mass @ actual[1:] + rhs[1:3], 0.0, atol=1.0e-6)
                    test.assertLessEqual(float(np.linalg.norm(actual[1:])), 0.300001)
                else:
                    np.testing.assert_allclose(actual, [1.0, *expected], atol=1.0e-6)


def test_sliding_direction(test, device):
    """Oppose sliding in every direction for analytic and mesh boxes, with and without patches."""
    for beta in (0.0, 0.2):
        for mesh in (False, True):
            for angle in (0.0, np.pi / 6, np.pi / 4, 2 * np.pi / 3):
                with test.subTest(beta=beta, mesh=mesh, angle=angle):
                    velocity, direction, overflow = _slide(device, beta, mesh, angle)
                    np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
                    test.assertFalse(overflow.any())


class TestSteelFrictionDirection(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _fn in (
    test_articulated_sliding_direction,
    test_locked_tangent_preserves_other_friction_direction,
    test_unequal_tangent_masses,
    test_sliding_direction,
):
    add_function_test(TestSteelFrictionDirection, _fn.__name__, _fn, devices=devices)


MODES = ("fixed_normal", "bisection", "bisection_desaxce", "coulomb_newton")


DT = 1.0 / 240.0


G = 9.81


def _box_on_slope(device, slope_angle, mu, articulated):
    """A box on the ground under gravity tilted by ``slope_angle``, free or on three prismatic joints."""
    builder = newton.ModelBuilder(gravity=(G * np.sin(slope_angle), 0.0, -G * np.cos(slope_angle)))
    builder.default_shape_cfg.mu = mu
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
    if articulated:
        link = builder.add_link(xform=xform)
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        axis = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            -1,
            link,
            linear_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Y), axis(axis=newton.Axis.Z)],
            parent_xform=xform,
        )
        builder.add_articulation([joint])
    else:
        body = builder.add_body(xform=xform)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _slide_friction_mode(device, mode, slope_angle, mu, articulated, steps=120):
    """Return the box's final body velocity and height."""
    model = _box_on_slope(device, slope_angle, mu, articulated)
    solver = SolverSteel(model, iterations=32, friction_mode=mode, friction_anchor_beta=0.0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return state_0.body_qd.numpy()[0], float(state_0.body_q.numpy()[0, 2])


def test_every_mode_holds_inside_the_cone(test, device):
    """Hold a free and an articulated box at rest inside the friction cone in every mode."""
    for articulated in (False, True):
        for mode in MODES:
            with test.subTest(articulated=articulated, mode=mode):
                velocity, height = _slide_friction_mode(device, mode, slope_angle=0.3, mu=0.5, articulated=articulated)
                test.assertLess(float(np.abs(velocity).max()), 1.0e-3)
                test.assertAlmostEqual(height, 0.1, delta=1.0e-3)


def test_sliding_follows_each_modes_coulomb_update(test, device):
    """Slide outside the friction cone: exact for current and coulomb_newton, approximate for bisection.

    ``g (sin a - mu cos a)`` is the exact sliding acceleration. The bisection modes solve the normal
    impulse to a fixed tolerance per sweep and slide slightly faster on free bodies; the de Saxce bias
    also lifts a sliding free body off the ground.
    """
    slope_angle, mu, steps = 0.6, 0.3, 120
    expected = G * (np.sin(slope_angle) - mu * np.cos(slope_angle)) * steps * DT
    results = {}
    for articulated in (False, True):
        for mode in MODES:
            results[(articulated, mode)] = _slide_friction_mode(device, mode, slope_angle, mu, articulated, steps)
    for articulated in (False, True):
        for mode in ("fixed_normal", "coulomb_newton"):
            with test.subTest(articulated=articulated, mode=mode):
                velocity, height = results[(articulated, mode)]
                test.assertAlmostEqual(float(velocity[0]), expected, delta=0.005 * expected)
                test.assertAlmostEqual(height, 0.1, delta=1.0e-3)
    for mode in ("bisection", "bisection_desaxce"):
        with test.subTest(articulated=True, mode=mode):
            velocity, height = results[(True, mode)]
            test.assertAlmostEqual(float(velocity[0]), expected, delta=0.005 * expected)
        with test.subTest(articulated=False, mode=mode):
            velocity, height = results[(False, mode)]
            test.assertGreater(float(velocity[0]), expected)
            test.assertAlmostEqual(float(velocity[0]), expected, delta=0.08 * expected)
    # The de Saxce bias raises the normal target velocity of a sliding free-body contact.
    test.assertAlmostEqual(results[(False, "bisection")][1], 0.1, delta=1.0e-4)
    test.assertGreater(results[(False, "bisection_desaxce")][1], 0.1 + 1.0e-3)
    test.assertGreater(float(results[(False, "bisection_desaxce")][0][2]), 1.0e-3)


def test_modes_select_distinct_kernels(test, device):
    """Build one matrix-free kernel per mode and keep the default kernel's name."""
    model = _box_on_slope(device, 0.3, 0.5, articulated=False)
    names = set()
    for mode in MODES:
        solver = SolverSteel(model, friction_mode=mode, friction_anchor_beta=0.0)
        name = solver._pgs_solve_mf_gs_kernel.key
        test.assertEqual(mode == "fixed_normal", not name.endswith(mode), name)
        names.add(name)
    test.assertEqual(len(names), len(MODES))


def test_unsupported_combinations_raise(test, device):
    """Reject the options the non-default modes do not implement."""
    model = _box_on_slope(device, 0.3, 0.5, articulated=True)
    with test.assertRaisesRegex(ValueError, "friction_mode"):
        SolverSteel(model, friction_mode="bad")
    for mode in MODES[1:]:
        with test.subTest(mode=mode):
            # An omitted friction_anchor_beta selects point friction, with a warning.
            with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "point friction"):
                solver = SolverSteel(model, friction_mode=mode)
            test.assertEqual(solver.friction_anchor_beta, 0.0)
            with test.assertRaisesRegex(ValueError, "friction_mode='fixed_normal'"):
                SolverSteel(model, friction_mode=mode, friction_anchor_beta=0.2)
            set_torsion_radius(model, 0.01)
            with test.assertRaisesRegex(ValueError, "friction_mode='fixed_normal'"):
                SolverSteel(model, friction_mode=mode, friction_anchor_beta=0.0)
            set_torsion_radius(model, 0.0)
            with test.assertRaisesRegex(ValueError, "friction_mode"):
                SolverSteel(model, friction_mode=mode, friction_anchor_beta=0.0, contact_compliance=True)
            with test.assertRaisesRegex(NotImplementedError, "friction_mode"):
                SolverSteel(model, friction_mode=mode, articulated_contact_response="propagation")


def test_sparse_factors_require_the_current_mode(test, device):
    """Select sparse mass factors only for the default friction update."""
    builder = newton.ModelBuilder()
    root = builder.add_link()
    builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
    joints = [builder.add_joint_revolute(-1, root, axis=newton.Axis.Z)]
    for side in (-1.0, 1.0):
        leg = builder.add_link()
        builder.add_shape_box(leg, hx=0.05, hy=0.05, hz=0.05)
        joints.append(
            builder.add_joint_revolute(
                root, leg, axis=newton.Axis.Y, parent_xform=wp.transform(wp.vec3(side * 0.2, 0, 0), wp.quat_identity())
            )
        )
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    test.assertIsNotNone(SolverSteel(model, friction_anchor_beta=0.0)._sparse_mass_matrix_size)
    for mode in MODES[1:]:
        with test.subTest(mode=mode):
            solver = SolverSteel(model, friction_mode=mode, friction_anchor_beta=0.0)
            test.assertIsNone(solver._sparse_mass_matrix_size)


def test_split_rejects_other_modes(test, device):
    """Reject the non-default modes in the split solve; the default omitted friction stays silent."""
    model = _box_on_slope(device, 0.3, 0.5, articulated=True)
    for mode in MODES[1:]:
        with test.subTest(mode=mode), warnings.catch_warnings():
            warnings.simplefilter("error")
            with test.assertRaisesRegex(NotImplementedError, f"friction_mode='{mode}'.*solve_mode='matrix_free'"):
                SolverSteel(model, solve_mode="split", friction_mode=mode)
    test.assertEqual(SolverSteel(model, solve_mode="split").friction_mode, "fixed_normal")


class TestSteelFrictionMode(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


for _fn in (
    test_every_mode_holds_inside_the_cone,
    test_sliding_follows_each_modes_coulomb_update,
    test_modes_select_distinct_kernels,
    test_unsupported_combinations_raise,
    test_sparse_factors_require_the_current_mode,
):
    add_function_test(TestSteelFrictionMode, _fn.__name__, _fn, devices=cuda_devices)


add_function_test(
    TestSteelFrictionMode,
    "test_split_rejects_other_modes",
    test_split_rejects_other_modes,
    devices=get_test_devices(),
)


def _box_on_support(device, mode, articulated, support_speed, steps=60, window=30, response="immediate"):
    """Step a box placed flat on a support moving at ``support_speed`` along x, with zero contact gap.

    A zero speed makes the support a static shape, otherwise a kinematic body. Returns the box's mean x
    velocity over the last ``window`` steps, its largest angular speed, the fewest contacts generated in a
    step and the matrix-free row count.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.0, restitution=0.0, density=500.0)
    support_pose = wp.transform((0.0, 0.0, -0.05), wp.quat_identity())
    if support_speed == 0.0:
        builder.add_shape_box(-1, xform=support_pose, hx=2.0, hy=2.0, hz=0.05, cfg=cfg)
    else:
        belt = builder.add_body(xform=support_pose)
        builder.add_shape_box(belt, hx=2.0, hy=2.0, hz=0.05, cfg=cfg)
        builder.body_flags[belt] = int(newton.BodyFlags.KINEMATIC)
    pose = wp.transform((0.0, 0.0, 0.05), wp.quat_identity())
    box = (builder.add_link if articulated else builder.add_body)(xform=pose)
    if articulated:
        joint = builder.add_joint_d6(
            parent=-1,
            child=box,
            parent_xform=pose,
            linear_axes=[newton.ModelBuilder.JointDofConfig(axis=axis) for axis in newton.Axis],
            angular_axes=[newton.ModelBuilder.JointDofConfig(axis=axis) for axis in newton.Axis],
        )
        builder.add_articulation([joint])
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05, cfg=cfg)
    model = builder.finalize(device=device)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        friction_mode=mode,
        friction_anchor_beta=0.0,
        iterations=32,
        free_body_constraint_max=64,
        articulation_constraint_max=64,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, deterministic=True)
    contacts = pipeline.contacts()
    state, next_state, control = model.state(), model.state(), model.control()
    if support_speed != 0.0:
        # The belt's free joint comes first.
        joint_qd = state.joint_qd.numpy()
        joint_qd[0] = support_speed
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    # Per-step history stays on the device until the end of the run.
    n = model.body_count
    body_q = wp.empty(steps * n, dtype=wp.transform, device=device)
    body_qd = wp.empty(steps * n, dtype=wp.spatial_vector, device=device)
    contact_count = wp.empty(steps, dtype=wp.int32, device=device)
    for step in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        wp.copy(contact_count, contacts.rigid_contact_count, dest_offset=step, count=1)
        solver.step(state, next_state, control, contacts, DT)
        state, next_state = next_state, state
        wp.copy(body_q, state.body_q, dest_offset=step * n, count=n)
        wp.copy(body_qd, state.body_qd, dest_offset=step * n, count=n)
    box_x = body_q.numpy().reshape(steps, n, 7)[:, box, 0]
    mean_vx = (box_x[-1] - box_x[-1 - window]) / (window * DT)
    max_spin = float(np.linalg.norm(body_qd.numpy().reshape(steps, n, 6)[:, box, 3:], axis=1).max())
    return mean_vx, max_spin, int(contact_count.numpy().min()), int(solver._mf_constraint_count.numpy().sum())


def test_kinematic_belt_drags_the_box(test, device):
    """Verify that a box on a kinematic belt reaches the belt speed in the coupled modes, free and articulated.

    Covers the subtests that fail when the sliding rows' rhs is ignored; articulated ``coulomb_newton``
    uses the fixed-normal pair and ``fixed_normal`` adds the rhs itself.
    """
    for articulated, modes in ((False, MODES[1:]), (True, ("bisection", "bisection_desaxce"))):
        for mode in modes:
            with test.subTest(articulated=articulated, mode=mode):
                mean_vx, _, _, mf_rows = _box_on_support(device, mode, articulated, support_speed=0.5)
                # The free box's contact is a matrix-free row; the articulated box's is a dense one.
                test.assertEqual(mf_rows == 0, articulated)
                test.assertAlmostEqual(mean_vx, 0.5, delta=5.0e-3)


class TestSteelKinematicBeltFriction(unittest.TestCase):
    pass


add_function_test(
    TestSteelKinematicBeltFriction,
    "test_kinematic_belt_drags_the_box",
    test_kinematic_belt_drags_the_box,
    devices=cuda_devices,
)


def test_resting_box_keeps_its_contacts(test, device):
    """Verify that a box resting on four zero-gap contacts stays still in the bisection modes, free and articulated.

    A box lifted by one float32 ulp loses its zero-gap contacts on the next step and falls, so this checks
    that the coupled normal solve reaches the support load as exactly as fixed_normal does.
    """
    for mode in ("bisection", "bisection_desaxce"):
        for articulated in (False, True):
            with test.subTest(mode=mode, articulated=articulated):
                _, max_spin, min_contacts, _ = _box_on_support(device, mode, articulated, support_speed=0.0)
                test.assertGreaterEqual(min_contacts, 4)
                test.assertLess(max_spin, 1.0e-6)


class TestSteelCoupledFrictionRest(unittest.TestCase):
    pass


add_function_test(
    TestSteelCoupledFrictionRest,
    "test_resting_box_keeps_its_contacts",
    test_resting_box_keeps_its_contacts,
    devices=cuda_devices,
)


@wp.kernel(enable_backward=False)
def _pose_motion_probe(
    before: wp.array[wp.transform],
    after: wp.array[wp.transform],
    point: wp.array[wp.vec3],
    result: wp.array[wp.vec3],
):
    i = wp.tid()
    result[i] = _pose_motion(after, before, i, point[i])


def _ground_box(device, **solver_kwargs):
    """Return a resting 1 kg box on the ground with patch friction enabled."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=125, mu=0.5))
    model = builder.finalize(device=device)
    kwargs = {"iterations": 32}
    kwargs.update(solver_kwargs)
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()

    def step(observables=None):
        s0, s1 = states
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, 0.005, observables=observables)
        states.reverse()

    return model, solver, contacts, step


@unittest.skipUnless(wp.is_cuda_available(), "SolverSteel requires CUDA")
class TestFrictionPatchHistory(unittest.TestCase):
    def test_support_edge_centers_ignore_contact_order(self):
        """Keep the same edge centers when contact ordering changes at rest."""
        device = _DEVICE
        points = np.array(
            [[x, y, 0.0] for x in np.linspace(-0.001, 0.001, 128) for y in (-0.05, 0.05)], dtype=np.float32
        )
        expected = np.array([[0.0, -0.05, 0.0], [0.0, 0.05, 0.0]], dtype=np.float32)
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                order = np.random.default_rng(seed).permutation(len(points))
                _, _, _, patches = _patch_fixture(points[order], device=device)
                locations = patches.view.point_a.numpy()[patches.view.weight.numpy() > 0.0]
                locations = locations[np.argsort(locations[:, 1])]
                # Summation order perturbs the principal axis only by roundoff.
                np.testing.assert_allclose(locations, expected, rtol=0.0, atol=1.0e-12)

    def test_narrow_patch_preserves_footprint_reflection_symmetry(self):
        """Avoid a diagonal friction couple on a symmetric narrow contact footprint."""
        device = _DEVICE
        for angle in (0.0, 0.37, 1.1):
            axis = np.array([np.sin(angle), np.cos(angle), 0.0])
            across = np.array([np.cos(angle), -np.sin(angle), 0.0])
            points = [x * across + y * axis for x in (-0.001, 0.001) for y in (-0.025, 0.025)]
            with self.subTest(angle=angle):
                _, _, _, patches = _patch_fixture(points, device=device)
                active = patches.view.weight.numpy() > 0.0
                locations = patches.view.point_a.numpy()[active]
                self.assertEqual(len(locations), 2)
                np.testing.assert_allclose(locations @ across, 0.0, atol=1.0e-7)
                np.testing.assert_allclose(np.sort(locations @ axis), [-0.025, 0.025], atol=1.0e-7)

    def test_sheared_narrow_patch_keeps_anchors_at_its_support_edges(self):
        """Keep both anchors near the support-edge centers when a slight shear tilts the principal axis."""
        device = _DEVICE
        # A faceted wheel's line contact: three points across each cap edge, one cap
        # offset slightly along the edges. The principal axis then deviates from the
        # caps' normal, and keeping only each edge's extreme point anchors opposite
        # corners, a diagonal couple that steers a symmetric wheel. Anchors on the
        # principal line through the centroid move only in proportion to the shear.
        across_offsets = (-0.0145, 0.0, 0.0145)
        for angle in (0.0, 0.37):
            axis = np.array([np.sin(angle), np.cos(angle), 0.0])
            across = np.array([np.cos(angle), -np.sin(angle), 0.0])
            for shear in (1.0e-6, 1.0e-5, -1.0e-4, 1.0e-3):
                points = [x * across - 0.025 * axis for x in across_offsets]
                points += [(x + shear) * across + 0.025 * axis for x in across_offsets]
                with self.subTest(angle=angle, shear=shear):
                    _, _, _, patches = _patch_fixture(points, device=device)
                    active = patches.view.weight.numpy() > 0.0
                    locations = patches.view.point_a.numpy()[active]
                    self.assertEqual(len(locations), 2)
                    locations = locations[np.argsort(locations @ axis)]
                    tolerance = abs(shear) + 1.0e-7
                    np.testing.assert_allclose(locations @ axis, [-0.025, 0.025], rtol=0.0, atol=tolerance)
                    np.testing.assert_allclose(locations @ across, [0.0, shear], rtol=0.0, atol=tolerance)

    def test_aligned_narrow_patch_anchors_span_the_full_footprint(self):
        """Anchor aligned narrow footprints at their axial extremes, including near-extreme samples."""
        device = _DEVICE
        theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        footprints = {
            # Edge samples just inside each end must not pull the anchors inward.
            "intermediate_edge_samples": [[x, y, 0.0] for x in (-0.001, 0.001) for y in (-0.025, -0.023, 0.023, 0.025)],
            "sampled_ellipse": np.stack([0.019 * np.cos(theta), 0.025 * np.sin(theta), np.zeros_like(theta)], axis=1),
        }
        for name, points in footprints.items():
            with self.subTest(footprint=name):
                _, _, _, patches = _patch_fixture(np.asarray(points, dtype=np.float32), device=device)
                active = patches.view.weight.numpy() > 0.0
                locations = patches.view.point_a.numpy()[active]
                self.assertEqual(len(locations), 2)
                locations = locations[np.argsort(locations[:, 1])]
                np.testing.assert_allclose(locations[:, 0], 0.0, atol=1.0e-7)
                np.testing.assert_allclose(locations[:, 1], [-0.025, 0.025], rtol=0.0, atol=1.0e-6)

    def test_narrow_patch_anchors_keep_each_end_height(self):
        """Anchor each end of a narrow footprint at the height of its own support edge."""
        device = _DEVICE
        # A faceted wheel bouncing on one cap touches at different heights at the
        # two ends. Level anchors there see opposite slip at the ends of a rolling
        # wheel, and the friction answers with a yaw kick.
        footprints = {
            "two_levels": ([[x, -0.025, 0.0] for x in (-0.0145, 0.0, 0.0145)] + [[0.0, 0.025, 0.001]], [0.0, 0.001]),
            "three_levels": ([[-0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.05, 0.0, 0.003]], [0.0, 0.003]),
        }
        for name, (members, heights) in footprints.items():
            with self.subTest(footprint=name):
                points = np.asarray(members, dtype=np.float32)
                _, _, _, patches = _patch_fixture(points, device=device)
                locations = patches.view.point_a.numpy()[patches.view.weight.numpy() > 0.0]
                self.assertEqual(len(locations), 2)
                axis = points[-1] - points[0]
                locations = locations[np.argsort(locations @ axis)]
                np.testing.assert_allclose(locations[:, 2], heights, rtol=0.0, atol=1.0e-7)

    def test_thin_triangle_anchor_span_varies_continuously(self):
        """Keep the anchor span continuous as a triangle thins into a collinear footprint."""
        device = _DEVICE
        spans = []
        for height in np.linspace(1.0e-6, 4.0e-6, 41):
            points = np.array([[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, height, 0.0]], dtype=np.float32)
            _, _, _, patches = _patch_fixture(points, device=device)
            locations = patches.view.point_a.numpy()[patches.view.weight.numpy() > 0.0]
            self.assertEqual(len(locations), 2)
            spans.append(np.ptp(locations[:, 0]))
        # A switch between two anchor constructions near collinearity would jump by tens of mm.
        self.assertLess(np.abs(np.diff(spans)).max(), 2.0e-3)

    def test_pose_increment_preserves_fixed_pivots_and_no_slip_rolling(self):
        """Distinguish rigid rotation from slip without querying the collision shape."""
        device = _DEVICE
        pivot = wp.vec3(0.3, -0.2, 0.5)
        origin = wp.vec3(-0.1, 0.2, 0.3)
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.7)
        before = [wp.transform(origin, wp.quat_identity()), wp.transform_identity()]
        after = [
            wp.transform(pivot + wp.quat_rotate(rotation, origin - pivot), rotation),
            wp.transform(wp.vec3(0.002, 0, 0), wp.quat_identity()),
        ]
        points = [pivot, wp.vec3(0)]
        for rate in (60, 120, 240):
            advance = 1.0 / rate
            # The rigid integrator advances translation and normalizes its
            # first-order quaternion update. Its no-slip rolling pair has
            # this relative rotation; a material chord falsely reports slip.
            angle = float(2.0 * np.arctan(0.5 * 20.0 / rate))
            before.append(wp.transform(wp.vec3(0, 0, 0.05), wp.quat_identity()))
            after.append(wp.transform(wp.vec3(advance, 0, 0.05), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), angle)))
            points.append(wp.vec3(advance, 0, 0))
        result = wp.zeros(len(points), dtype=wp.vec3, device=device)
        wp.launch(
            _pose_motion_probe,
            dim=len(points),
            inputs=[
                wp.array(before, dtype=wp.transform, device=device),
                wp.array(after, dtype=wp.transform, device=device),
                wp.array(points, dtype=wp.vec3, device=device),
                result,
            ],
            device=device,
        )
        values = result.numpy()
        np.testing.assert_allclose(values[0], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(values[1], [0.002, 0, 0], atol=1.0e-7)
        np.testing.assert_allclose(values[2:, :2], 0.0, atol=1.0e-7)

    def test_supported_normal_turn_preserves_friction_history(self):
        """Retain supported history during small relative rotations on either body."""
        for body in (0, 1):
            for axis, retained in ((wp.vec3(0, 1, 0), True), (wp.vec3(0, 0, 1), True)):
                with self.subTest(body=body, axis=axis):
                    model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
                    types = model.shape_type.numpy()
                    types[body] = int(newton.GeoType.SPHERE)
                    model.shape_type.assign(types)
                    patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
                    patches.store(state)
                    patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
                    self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
                    transforms = [wp.transform_identity(), wp.transform_identity()]
                    transforms[body] = wp.transform(wp.vec3(0), wp.quat_from_axis_angle(axis, 0.01))
                    state.body_q.assign(transforms)
                    patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
                    self.assertEqual(int(patches.current.source.numpy()[0]) >= 0, retained)

    def test_normal_turn_transports_tangent_error_without_decay(self):
        """Preserve spring length during supported rocking without additional slip."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        patches.store(state)
        pivot = wp.vec3(0.0005, 0, 0)
        local_point = wp.vec3(-0.0005, 0, 0)
        state.body_q.assign([wp.transform(wp.vec3(0.001, 0, 0), wp.quat_identity()), wp.transform_identity()])
        contacts.rigid_contact_point0.assign([local_point])
        contacts.rigid_contact_point1.assign([pivot])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        for step in range(100):
            patches.store(state)
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.025 if step % 2 else -0.025)
            state.body_q.assign(
                [wp.transform(pivot - wp.quat_rotate(rotation, local_point), rotation), wp.transform_identity()]
            )
            patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
            self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
            self.assertAlmostEqual(float(np.linalg.norm(patches.view.phi.numpy()[0])), 0.001, delta=1.0e-7)

    def test_shared_pad_randomization_preserves_material_regions(self):
        """Pool one sampled pad material while keeping genuinely different coefficients separate."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]], shape0=[0, 2])
        for coefficient in np.random.default_rng(42).uniform(0.1, 1.0, 3):
            model.shape_material_mu.assign([coefficient, 0.5, coefficient])
            patches.update_geometry(model)
            patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
            self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
            np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])
            patches.store(state)
        model.shape_material_mu.assign([0.2, 0.5, 0.8])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [1, 1])

    def test_pose_history_covers_bodies_beyond_contact_capacity(self):
        """Capture every body pose even when the contact buffer is smaller."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]], shape_bodies=(9, 10, 9))
        transforms = state.body_q.numpy()
        transforms[10, 0] = 0.002
        state.body_q.assign(transforms)
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        patches.store(state)
        transforms[10, 0] = 0.003
        state.body_q.assign(transforms)
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
        self.assertAlmostEqual(float(np.linalg.norm(patches.view.phi.numpy()[0])), 0.001, delta=1.0e-7)

    def test_rocking_face_retires_the_lifted_anchor(self):
        """Keep the supported edge's history and replace the opposite anchor after rocking."""
        points = [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]]
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 60.0)
        pivot = wp.vec3(-0.1, 0, 0)
        state.body_q.assign([wp.transform(pivot - wp.quat_rotate(rotation, pivot), rotation), wp.transform_identity()])
        contacts.rigid_contact_count.assign([2])
        edge = [points[0], points[2], points[0], points[2]]
        contacts.rigid_contact_point0.assign(edge)
        contacts.rigid_contact_point1.assign(edge)
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        active = patches.current.valid.numpy() != 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_allclose(patches.current.anchor_a.numpy()[active, 0], -0.1, atol=1.0e-7)
        self.assertEqual(np.count_nonzero(patches.current.source.numpy()[active] >= 0), 1)

    def test_unloading_penetration_keeps_supported_anchors(self):
        """Retain history while decompression leaves the actual surfaces in contact."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_point0.assign([[-0.1, 0, -0.005], [0.1, 0, -0.005]])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        patches.store(state)
        state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.003), wp.quat_identity()), wp.transform_identity()])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 2)

    def test_support_witnesses_use_existing_contact_gap_limits(self):
        """Keep supported history within the contact envelope and respect tighter solver gates."""
        for limits, carried in (
            ({}, True),
            ({"contact_gap_limit": 0.0001}, False),
            ({"friction_gap_limit": 0.0001}, True),
        ):
            with self.subTest(limits=limits):
                model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
                model.shape_gap.fill_(0.0005)
                patches.store(state)
                state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.0002), wp.quat_identity()), wp.transform_identity()])
                contacts.rigid_contact_point0.assign([[-0.1, 0, -0.0004], [0.1, 0, -0.0004]])
                patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu, **limits)
                self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 2 if carried else 0)
                # A fresh sample on the supported region cannot retain history
                # whose old footprint has lifted beyond the shape gap envelope.
                state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.0012), wp.quat_identity()), wp.transform_identity()])
                contacts.rigid_contact_point0.assign([[-0.1, 0, -0.0014], [0.1, 0, -0.0014]])
                patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu, **limits)
                self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 0)

    def test_geometry_edits_retire_only_affected_history(self):
        """Invalidate edited surfaces across carrier seams while preserving unrelated patches."""
        for field in ("shape_transform", "shape_scale", "shape_margin", "shape_source_ptr"):
            for static in (False, True):
                with self.subTest(field=field, static=static):
                    model, state, contacts, patches = _patch_fixture(
                        [[0, 0, 0], [0, 0, 0]],
                        shape0=[0, 3],
                        shape_bodies=(0, -1 if static else 1, 0, 2, 3),
                        materials=(0.5,) * 5,
                    )
                    contacts.rigid_contact_shape1.assign([1, 4])
                    patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
                    patches.store(state)
                    np.testing.assert_array_equal(patches.previous.valid.numpy()[:2], [1, 1])
                    # Static geometry is identified by shape. A dynamic edit on
                    # shape 2 must also retire shape 0's anchor on the same body.
                    edited = 1 if static else 2
                    values = getattr(model, field).numpy()
                    if field == "shape_transform":
                        values[edited, 0] += 1
                    else:
                        values[edited] += 1
                    getattr(model, field).assign(values)
                    patches.update_geometry(model)
                    np.testing.assert_array_equal(patches.previous.valid.numpy()[:2], [0, 1])

    def test_geometry_updates_allow_contacts_without_a_shape(self):
        """Preserve implicit static contacts when an unrelated shape changes."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]], shape_bodies=(0, 1, 1))
        contacts.rigid_contact_shape1.assign([-1])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        patches.store(state)
        self.assertEqual(int(patches.previous.valid.numpy()[0]), 1)
        # Editing the last shape must not alias the contact's -1 sentinel.
        transforms = model.shape_transform.numpy()
        transforms[-1, 0] = 1.0
        model.shape_transform.assign(transforms)
        patches.update_geometry(model)
        self.assertEqual(int(patches.previous.valid.numpy()[0]), 1)

    def test_reused_contacts_keep_anchors_across_substeps(self):
        """Accumulate slip across substeps regardless of stale collision match indices."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_match_index = wp.array([1, 0], dtype=int, device=model.device)
        active = patches.current.valid.numpy() != 0
        patches.store(state)
        for substep in range(1, 4):
            displacement = substep * 0.001
            state.body_q.assign(
                [wp.transform(wp.vec3(displacement, 0, 0), wp.quat_identity()), wp.transform_identity()]
            )
            patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
            active = patches.current.valid.numpy() != 0
            np.testing.assert_allclose(
                np.linalg.norm(patches.view.phi.numpy()[active], axis=1), displacement, atol=1.0e-7
            )
            self.assertTrue(np.all(patches.current.source.numpy()[active] >= 0))
            patches.store(state)

    def test_connected_convex_chain_forms_one_region(self):
        """Join a chain into one friction region even when the two end shapes do not overlap."""
        model, state, contacts, patches = _patch_fixture(
            [[-0.15, 0, 0], [0.15, 0, 0], [-0.05, 0, 0], [0.05, 0, 0]],
            shape0=[0, 4, 2, 3],
            shape_bodies=(0, 1, 0, 0, 0),
            materials=(0.5,) * 5,
        )
        model.shape_transform.assign(
            [wp.transform(wp.vec3(x, 0, 0), wp.quat_identity()) for x in (-0.15, 0, -0.05, 0.05, 0.15)]
        )
        model.shape_collision_radius.assign([0.06, 0.3, 0.06, 0.06, 0.06])
        patches.update_geometry(model)
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:4], [0.5, 0.5, 0, 0])

    def test_disconnected_convex_shapes_form_separate_regions(self):
        """Keep two separated pads on one body from pooling friction through empty space."""
        model, state, contacts, patches = _patch_fixture([[-0.15, 0, 0], [0.15, 0, 0]], shape0=[0, 2])
        model.shape_transform.assign(
            [
                wp.transform(wp.vec3(-0.15, 0, 0), wp.quat_identity()),
                wp.transform_identity(),
                wp.transform(wp.vec3(0.15, 0, 0), wp.quat_identity()),
            ]
        )
        model.shape_collision_radius.assign([0.03, 0.2, 0.03])
        patches.update_geometry(model)
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [1, 1])

    def test_incomplete_frame_cannot_create_anchor_history(self):
        """Reject an overflowing contact frame before selecting or carrying anchors."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_count.assign([3])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        self.assertEqual(np.count_nonzero(patches.current.valid.numpy()), 0)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 0)

    def test_reversed_contact_orientation_uses_same_patch(self):
        """Give a body pair one identity regardless of collision shape ordering."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_shape0.assign([0, 1])
        contacts.rigid_contact_shape1.assign([1, 0])
        contacts.rigid_contact_normal.assign([[0, 0, -1], [0, 0, 1]])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])

    def test_saturation_only_releases_an_anchor_with_sliding_motion(self):
        """Keep history under impending slip, and release it under motion against saturated friction."""
        _, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        zeros = wp.zeros(1, dtype=int, device=_DEVICE)
        lengths = wp.array([3], dtype=int, device=_DEVICE)
        parents = wp.array([[0, 0, 0]], dtype=int, device=_DEVICE)
        mu = wp.array([[0.5, 0.5, 0.5]], dtype=float, device=_DEVICE)
        impulses = wp.array([[2, 1, 0]], dtype=float, device=_DEVICE)
        velocity = wp.zeros(2, dtype=wp.spatial_vector, device=_DEVICE)
        args = [
            contacts.rigid_contact_count,
            patches.current,
            state.body_q,
            state.body_q,
            velocity,
            wp.zeros(2, dtype=wp.vec3, device=_DEVICE),
            zeros,
            zeros,
            zeros,
            lengths,
            0,
            parents,
            mu,
            impulses,
            0.005,
        ]
        impulses.zero_()
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device=_DEVICE)
        self.assertEqual(patches.current.valid.numpy()[0], 0, "unloaded speculative contacts must not accrue stiction")
        patches.current.valid.fill_(1)
        impulses.assign([[2, 1, 0]])
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device=_DEVICE)
        self.assertEqual(patches.current.valid.numpy()[0], 1)
        velocity.assign([[0, -0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device=_DEVICE)
        self.assertEqual(patches.current.valid.numpy()[0], 0)

    def test_convex_shapes_share_one_body_patch(self):
        """Pool friction load across adjacent convex pieces with compatible materials."""
        _, _, _, patches = _patch_fixture(
            [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], shape0=[0, 2, 0, 2]
        )
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 2)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        self.assertAlmostEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_material_normal_and_disconnected_regions_stay_separate(self):
        """Keep friction from pooling across incompatible or spatially separate regions."""
        cases = (
            {"points": [[0, 0, 0], [0.01, 0, 0]], "shape0": [0, 2], "materials": (0.5, 0.5, 0.8)},
            {"points": [[0, 0, 0], [0.01, 0, 0]], "normals": [[0, 0, -1], [0, -1, 0]]},
            {"points": [[0, 0, 0], [0, 0, 0.05]]},
            {"points": [[0, 0, 0], [1, 0, 0]]},
        )
        for case in cases:
            with self.subTest(case=case):
                _, _, _, patches = _patch_fixture(**case)
                self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
                np.testing.assert_array_equal(patches.view.weight.numpy()[:2], [1, 1])

    def test_contact_churn_preserves_error_at_current_points(self):
        """Preserve a sticking region's history even when every contact sample changes."""
        points = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], dtype=np.float32)
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        points = points[[3, 1, 0, 2]] * 0.98
        contacts.rigid_contact_point0.assign(points)
        contacts.rigid_contact_point1.assign(points)
        state.body_q.assign([wp.transform(wp.vec3(0.001, 0, 0), wp.quat_identity()), wp.transform_identity()])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        active = patches.current.valid.numpy() != 0
        np.testing.assert_allclose(patches.view.point_a.numpy()[active], patches.current.center.numpy()[active])
        np.testing.assert_allclose(patches.view.point_b.numpy()[active], patches.current.center.numpy()[active])
        np.testing.assert_allclose(np.linalg.norm(patches.view.phi.numpy()[active], axis=1), 0.001, atol=1.0e-7)
        self.assertTrue(np.all(patches.current.source.numpy()[active] >= 0))

    def test_contact_migration_transports_twist_to_new_lever_arms(self):
        """Preserve a region's twist rather than copying error from old lever arms."""
        points = np.array([[-0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.01)
        state.body_q.assign([wp.transform(wp.vec3(0), rotation), wp.transform_identity()])
        contacts.rigid_contact_point0.assign([wp.quat_rotate(wp.quat_inverse(rotation), wp.vec3(*p)) for p in points])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        before = patches.view.phi.numpy()[:2].copy()
        self.assertGreater(np.linalg.norm(before), 0.001)
        patches.store(state)
        points *= 0.8
        contacts.rigid_contact_point0.assign([wp.quat_rotate(wp.quat_inverse(rotation), wp.vec3(*p)) for p in points])
        contacts.rigid_contact_point1.assign(points)
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        np.testing.assert_allclose(patches.view.phi.numpy()[:2], 0.8 * before, atol=1.0e-7)
        self.assertTrue((patches.current.source.numpy()[:2] >= 0).all())

    def test_coincident_contacts_use_one_anchor(self):
        """Reject duplicate witnesses that would create two coincident friction constraints."""
        _, _, _, patches = _patch_fixture([[0, 0, 0]] * 4)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 1)
        self.assertEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_warmstart_uses_patch_load_when_anchor_normal_is_unloaded(self):
        """Keep a cached tangent supported when other normals carry the load."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]])
        patches.current.tangent_impulse.assign([[0.8, 0, 0], [0, 0, 0], [0.8, 0, 0]] + [[0, 0, 0]] * 5)
        patches.store(state)
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        slots = wp.array([0, 3, 4], dtype=int, device=_DEVICE)
        worlds = wp.zeros(3, dtype=int, device=_DEVICE)
        paths = wp.zeros(3, dtype=int, device=_DEVICE)
        lengths = wp.array([3, 1, 3], dtype=int, device=_DEVICE)
        parents = wp.array([[-1, 0, 0, -1, -1, 4, 4]], dtype=int, device=_DEVICE)
        mu = wp.array([[0.5] * 7], dtype=float, device=_DEVICE)
        impulses = wp.array([[0, 0, 0, 4, 0, 0, 0]], dtype=float, device=_DEVICE)
        wp.launch(
            link_patch_rows,
            dim=3,
            inputs=[contacts.rigid_contact_count, patches.view, worlds, slots, paths, lengths, 0, parents, mu],
            device=_DEVICE,
        )
        wp.launch(
            seed_patch_impulses,
            dim=3,
            inputs=[
                contacts.rigid_contact_count,
                patches.current,
                patches.previous,
                state.body_q,
                worlds,
                slots,
                paths,
                lengths,
                0,
                parents,
                mu,
                impulses,
                1.0,
                1.0,
                wp.zeros(1, dtype=float, device=_DEVICE),
            ],
            device=_DEVICE,
        )
        result = impulses.numpy()[0]
        self.assertAlmostEqual(float(np.linalg.norm(result[1:3])), 0.8)
        self.assertAlmostEqual(float(np.linalg.norm(result[5:7])), 0.8)

    def test_seed_keeps_matched_warmstart_without_patch_history(self):
        """Keep the contact-matched friction warm start on an anchor that has no patch history yet."""
        _model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        self.assertEqual(int(patches.current.source.numpy()[0]), -1)
        slots = wp.zeros(1, dtype=int, device=_DEVICE)
        lengths = wp.array([3], dtype=int, device=_DEVICE)
        parents = wp.array([[-1] * 8], dtype=int, device=_DEVICE)
        mu = wp.array([[0.5, 0.25, 0.25] + [0.0] * 5], dtype=float, device=_DEVICE)
        seeded = [1.0, 0.3, -0.2] + [0.0] * 5
        impulses = wp.array([seeded], dtype=float, device=_DEVICE)
        wp.launch(
            seed_patch_impulses,
            dim=1,
            inputs=[
                contacts.rigid_contact_count,
                patches.current,
                patches.previous,
                state.body_q,
                slots,
                slots,
                slots,
                lengths,
                0,
                parents,
                mu,
                impulses,
                1.0,
                1.0,
                wp.zeros(1, dtype=float, device=_DEVICE),
            ],
            device=_DEVICE,
        )
        np.testing.assert_array_equal(impulses.numpy()[0], np.array(seeded, dtype=np.float32))

    def test_anchors_without_rows_keep_history(self):
        """Keep a valid anchor and its cached impulse when its region gets no friction rows this step."""
        _model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        cached = [[0.1, 0.2, 0.0]] + [[0.0, 0.0, 0.0]] * 7
        zeros = wp.zeros(1, dtype=int, device=_DEVICE)
        parents = wp.array([[-1] * 8], dtype=int, device=_DEVICE)
        mu = wp.zeros((1, 8), dtype=float, device=_DEVICE)
        impulses = wp.zeros((1, 8), dtype=float, device=_DEVICE)
        qd = wp.zeros(2, dtype=wp.spatial_vector, device=_DEVICE)
        com = wp.zeros(2, dtype=wp.vec3, device=_DEVICE)
        for slot, length, load, retained in ((-1, 3, 0.0, 1), (0, 1, 2.0, 1), (0, 1, 0.0, 0)):
            with self.subTest(slot=slot, slots_needed=length):
                impulses.assign([[load] + [0.0] * 7])
                patches.current.valid.assign([1] + [0] * 7)
                patches.current.tangent_impulse.assign(cached)
                wp.launch(
                    finish_patch_impulses,
                    dim=1,
                    inputs=[
                        contacts.rigid_contact_count,
                        patches.current,
                        state.body_q,
                        state.body_q,
                        qd,
                        com,
                        zeros,
                        wp.array([slot], dtype=int, device=_DEVICE),
                        zeros,
                        wp.array([length], dtype=int, device=_DEVICE),
                        0,
                        parents,
                        mu,
                        impulses,
                        0.005,
                    ],
                    device=_DEVICE,
                )
                self.assertEqual(int(patches.current.valid.numpy()[0]), retained)
                np.testing.assert_array_equal(
                    patches.current.tangent_impulse.numpy()[0], np.asarray(cached[0], dtype=np.float32)
                )

    def test_stationary_history_does_not_accumulate_roundoff(self):
        """Avoid accumulating world round-trip error while a transformed contact remains stationary."""
        model, state, contacts, patches = _patch_fixture([[0.02, -0.03, 0.0]])
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.5, 0.8)), 1.1)
        state.body_q.assign([wp.transform(wp.vec3(1.3, -0.7, 2.1), rotation), wp.transform_identity()])
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
        for _ in range(100):
            patches.store(state)
            patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu)
            self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
            # Each step's world round trip may round in float32 on the device, but the
            # carried displacement must not grow with the number of stationary steps.
            np.testing.assert_allclose(patches.view.phi.numpy()[0], [0.0, 0.0], atol=1.0e-7)

    def test_filtered_members_carry_history_without_starting_it(self):
        """Carry existing anchors through a friction-filtered step with zero weight, and never create new ones."""
        model, state, contacts, patches = _patch_fixture([[-0.05, 0, 0], [0.05, 0, 0]])
        patches.store(state)
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu, friction_gap_limit=-1.0)
        self.assertEqual(int(patches.current.valid.numpy().sum()), 2)
        self.assertTrue((patches.current.source.numpy()[:2] >= 0).all())
        self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)
        patches.previous.valid.zero_()
        patches.build(model, state, contacts, shape_material_mu=model.shape_material_mu, friction_gap_limit=-1.0)
        self.assertEqual(int(patches.current.valid.numpy().sum()), 0)
        self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)


def _run_rolling(
    geometry, device, launches=((0.0, 1, None),), *, friction_anchor_beta=None, segments=64, hz=240, deterministic=False
):
    """Roll one generated wheel per world for one second and average each world's speed over its final quarter.

    Each launch is a ``(phase, direction, mu)`` world: ``phase`` turns the wheel about its axle [rad] before launch,
    which selects the facet it first lands on, and ``mu`` overrides the default friction coefficient unless ``None``.
    Returns ``(pose, velocity, mean_velocity)`` per launch.
    """
    builder = newton.ModelBuilder()
    mesh = newton.Mesh.create_cylinder(radius=0.05, half_height=0.025, segments=segments)
    for phase, _direction, mu in launches:
        shape_kwargs = {} if mu is None else {"cfg": newton.ModelBuilder.ShapeConfig(mu=mu)}
        builder.begin_world()
        builder.add_ground_plane(**shape_kwargs)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(phase))
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.05), rotation))
        if geometry == "sphere":
            builder.add_shape_sphere(body, radius=0.05, **shape_kwargs)
        elif geometry in ("mesh", "convex_hull"):
            getattr(builder, "add_shape_" + geometry)(body, mesh=mesh, **shape_kwargs)
        else:
            rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)
            add_shape = getattr(builder, "add_shape_" + geometry)
            add_shape(body, radius=0.05, half_height=0.025, xform=wp.transform(wp.vec3(0), rotation), **shape_kwargs)
        builder.end_world()
    model = builder.finalize(device=device)
    kwargs = {} if friction_anchor_beta is None else {"friction_anchor_beta": friction_anchor_beta}
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **kwargs)
    pipeline = newton.CollisionPipeline(model, deterministic=deterministic)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    s0.joint_qd.assign(np.ravel([[direction, 0, 0, 0, 20 * direction, 0] for _phase, direction, _mu in launches]))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    control = model.control()
    final_velocity = []
    for step in range(hz):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, 1.0 / hz)
        s0, s1 = s1, s0
        if step >= 3 * hz // 4:
            # CPU arrays expose their buffer; copy before the states swap.
            final_velocity.append(s0.body_qd.numpy().copy())
    poses, velocities, mean_velocities = s0.body_q.numpy(), s0.body_qd.numpy(), np.mean(final_velocity, axis=0)
    return [(poses[i], velocities[i], mean_velocities[i]) for i in range(len(launches))]


@unittest.skipUnless(wp.is_cuda_available(), "SolverSteel requires CUDA")
class TestSteelFrictionRolling(unittest.TestCase):
    """Rolling, resting and twisting bodies under default patch friction."""

    def test_default_friction_preserves_free_rolling(self):
        """Match velocity-only free rolling without adding a persistent rearward friction force."""
        device = _DEVICE
        for geometry in ("sphere", "capsule", "cylinder", "mesh", "convex_hull"):
            results = []
            # Isolate displacement persistence from row reduction on faceted
            # wheels: a negligible positive gain keeps the same patch rows with
            # effectively no positional correction. The separate tessellation
            # matrix compares default patches against full point friction.
            reference_beta = 1.0e-8 if geometry in ("mesh", "convex_hull") else 0.0
            # At 240 Hz the 64-facet wheel advances almost one facet per step.
            # Resolve facet crossings for this strict persistence comparison;
            # the tessellation matrix below retains the coarse-timestep cases.
            hz = 1920 if geometry in ("mesh", "convex_hull") else 240
            for kwargs in ({"friction_anchor_beta": reference_beta}, {}):
                (result,) = _run_rolling(geometry, device, hz=hz, deterministic=True, **kwargs)
                self.assertTrue(all(np.isfinite(value).all() for value in result))
                results.append(result)
            with self.subTest(geometry=geometry):
                reference, actual = results
                self.assertAlmostEqual(float(actual[0][0]), float(reference[0][0]), delta=0.01)
                # Patch rows on a faceted wheel wobble laterally (the point-friction wheel stays within
                # 0.3 mm); keep the lateral error inside the tessellation matrix's 3% travel bound.
                lateral_bound = 0.02 if geometry in ("mesh", "convex_hull") else 0.01
                self.assertLess(abs(float(actual[0][1] - reference[0][1])), lateral_bound)
                # Facet impacts lose energy even without positional correction;
                # compare averaged speed rather than individual impact phases.
                self.assertAlmostEqual(float(actual[2][0]), float(reference[2][0]), delta=0.02)
                self.assertAlmostEqual(float(actual[2][4]), float(reference[2][4]), delta=0.4)
                self.assertLess(abs(float(actual[2][1])), 0.01)
                if geometry not in ("mesh", "convex_hull"):
                    self.assertAlmostEqual(float(actual[1][0]), float(reference[1][0]), delta=0.01)
                    self.assertAlmostEqual(float(actual[1][4]), float(reference[1][4]), delta=0.2)
                    self.assertGreater(float(actual[1][0]), 0.9)
                    self.assertLess(abs(float(actual[1][0] - 0.05 * actual[1][4])), 0.01)

    def test_faceted_rolling_across_tessellation_direction_and_timestep(self):
        """Bound patch approximation error across coarse and fine faceted wheels."""
        device = _DEVICE
        # A coarse-timestep wheel bounces from facet to facet, so one trajectory is
        # chaotic: the starting facet, or a round-off change in contact order, moves
        # its one-second travel by centimetres, for point friction as much as for
        # patches. Fix the contact order and compare the means over evenly spaced
        # starting facet phases, so each case checks the approximation rather than
        # one draw of that spread. Phases and directions run as worlds of one model.
        phase_count = 4
        directions = (-1, 1)
        for geometry in ("mesh", "convex_hull"):
            for segments in (32, 64, 128):
                phases = [2.0 * np.pi / segments * k / phase_count for k in range(phase_count)]
                launches = [(phase, direction, None) for direction in directions for phase in phases]
                # Frictionless wheels launched forward must fail the same travel bound.
                controls = [(phase, 1, 0.0) for phase in phases]
                for hz in (120, 240):
                    case = {"segments": segments, "hz": hz, "deterministic": True}
                    references = _run_rolling(geometry, device, launches, friction_anchor_beta=0.0, **case)
                    actuals = _run_rolling(geometry, device, launches + controls, **case)
                    for i, direction in enumerate(directions):
                        runs = slice(i * phase_count, (i + 1) * phase_count)
                        reference = [np.mean([r[k] for r in references[runs]], axis=0) for k in range(3)]
                        actual = [np.mean([a[k] for a in actuals[runs]], axis=0) for k in range(3)]
                        with self.subTest(geometry=geometry, segments=segments, hz=hz, direction=direction):
                            for result in actuals[runs]:
                                self.assertTrue(all(np.isfinite(value).all() for value in result))
                            # A two-location friction wrench approximates the
                            # per-contact reference. Bound its one-second travel
                            # error to 3% and mean speed error to 6% of launch speed;
                            # include lateral motion so steering errors cannot hide.
                            self.assertLess(np.max(np.abs(actual[0][:2] - reference[0][:2])), 0.03)
                            self.assertLess(abs(float(actual[2][0] - reference[2][0])), 0.06)
                            # Facet rocking can change the phase of lateral
                            # oscillations; bound their absolute speed as well
                            # as the lateral displacement checked above.
                            self.assertLess(abs(float(actual[2][1])), 0.05)
                            # Same 6% margin on the 20 rad/s launch spin.
                            self.assertLess(abs(float(actual[2][4] - reference[2][4])), 1.2)
                        if direction == 1:
                            control = np.mean([a[0] for a in actuals[-phase_count:]], axis=0)
                            with self.subTest(geometry=geometry, segments=segments, hz=hz, control="frictionless"):
                                self.assertGreater(np.max(np.abs(control[:2] - reference[0][:2])), 0.03)

    def test_default_friction_keeps_a_box_stack_at_rest(self):
        """Settle an ordinary stack without grasp-specific settings or anchor opt-in."""
        device = _DEVICE
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for height in (0.05, 0.151, 0.252):
            body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, height), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free")
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        for _ in range(240):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1.0 / 240.0)
            s0, s1 = s1, s0
        poses, velocities = s0.body_q.numpy(), s0.body_qd.numpy()
        self.assertTrue(np.isfinite(poses).all() and np.isfinite(velocities).all())
        np.testing.assert_allclose(poses[:, 2], [0.05, 0.15, 0.25], atol=0.005)
        self.assertLess(float(np.max(np.abs(poses[:, :2]))), 0.005)
        self.assertLess(float(np.max(np.linalg.norm(velocities[:, :3], axis=1))), 0.05)
        self.assertGreater(np.count_nonzero(solver._friction_patches.previous.valid.numpy()), 0)

    def test_patch_resists_twist_and_releases_above_its_limit(self):
        """Resist a static yaw torque with separated anchors, but permit a larger torque to spin."""
        device = _DEVICE
        for torque in (0.2, 1.0):
            with self.subTest(torque=torque), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
                body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
                builder.add_shape_box(
                    body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=125, mu=0.5)
                )
                model = builder.finalize()
                solver = newton.solvers.SolverSteel(
                    model, solve_mode="matrix_free", friction_anchor_beta=0.2, iterations=64
                )
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
                contacts = pipeline.contacts()
                s0, s1 = model.state(), model.state()
                control = model.control()
                for _ in range(100):
                    s0.body_f.assign([[0, 0, 0, 0, 0, torque]])
                    pipeline.collide(s0, contacts)
                    solver.step(s0, s1, control, contacts, 0.005)
                    s0, s1 = s1, s0
                angular_speed = abs(float(s0.body_qd.numpy()[0, 5]))
                self.assertTrue(np.isfinite(s0.body_q.numpy()).all())
                if torque == 0.2:
                    self.assertLess(angular_speed, 0.01)
                    self.assertLess(abs(float(s0.body_q.numpy()[0, 5])), 0.01)
                else:
                    self.assertGreater(angular_speed, 1.0)


@unittest.skipUnless(wp.is_cuda_available(), "SolverSteel requires CUDA")
class TestSteelFrictionGraspHistory(unittest.TestCase):
    """Patch history through grasps, geometry edits, resets and friction notifications."""

    def test_curved_grasp_preserves_history_under_small_disturbances(self):
        """Bound held sphere/capsule drift despite repeated small relative rotations."""
        device = _DEVICE
        with wp.ScopedDevice(device):
            for geometry in ("sphere", "capsule"):
                drifts = []
                for enabled in (False, True):
                    model, jaws, obj = _build_v_jaws(5.0, geometry)
                    # Four iterations leave point friction under-converged on curved contacts.
                    kwargs = dict(_SQUEEZE_SOLVER, iterations=4)
                    if not enabled:
                        kwargs["friction_anchor_beta"] = 0.0
                    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **kwargs)
                    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn")
                    contacts = pipeline.contacts()
                    s0, s1 = model.state(), model.state()
                    control = model.control()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
                    relative_height = []
                    retained = []
                    for step in range(600):
                        if step >= 100:
                            # External pose disturbances exercise history through
                            # normal changes, even when stiction resists the torque.
                            q = s0.joint_q.numpy()
                            angle = 2.0e-4 * (np.sin(step * 0.07) - np.sin((step - 1) * 0.07))
                            q[-4:] = np.asarray(
                                wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(angle)) * wp.quat(*q[-4:])
                            )
                            s0.joint_q.assign(q)
                            newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
                        pipeline.collide(s0, contacts)
                        s0.clear_forces()
                        force = s0.body_f.numpy()
                        force[obj, 4] = 2.0e-4 * np.sin(step * 0.07)
                        s0.body_f.assign(force)
                        solver.step(s0, s1, control, contacts, 0.005)
                        s0, s1 = s1, s0
                        pose = s0.body_q.numpy()
                        self.assertTrue(np.isfinite(pose).all())
                        relative_height.append(pose[obj, 2] - pose[jaws[0], 2])
                        if enabled and step >= 100:
                            retained.append(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0) >= 2)
                    drifts.append(float(np.ptp(relative_height[100:])))
                    if enabled:
                        with self.subTest(geometry=geometry):
                            self.assertGreater(np.mean(retained), 0.8)
                            self.assertLess(drifts[-1], 1.0e-4)
                with self.subTest(geometry=geometry):
                    # Anchoring must at least halve the anchor-free drift.
                    self.assertGreater(drifts[0], 2 * drifts[1])

    def test_default_patch_friction_and_explicit_opt_out(self):
        """Build persistent patches by default while retaining an explicit velocity-only opt-out."""
        device = _DEVICE
        for kwargs, enabled in (({}, True), ({"friction_anchor_beta": 0.0}, False)):
            with self.subTest(enabled=enabled):
                _, solver, _, step = _ground_box(device, **kwargs)
                for _ in range(3):
                    step()
                self.assertEqual(solver._friction_anchors_enabled, enabled)
                if enabled:
                    self.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
                    self.assertEqual(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0), 2)
                else:
                    self.assertFalse(hasattr(solver._friction_patches, "fixed_normal"))

    def test_rocking_cube_assigns_friction_only_to_supported_edge(self):
        """Retire the lifted face anchor before building rows for a cube resting on one edge."""
        device = _DEVICE
        builder = newton.ModelBuilder()
        builder.rigid_gap = 1.0e-4
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverSteel(
            model, solve_mode="matrix_free", friction_anchor_beta=0.2, contact_gap_gate=1.0e-4
        )
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts, 0.005)
        self.assertEqual(np.count_nonzero(solver._friction_patches.previous.valid.numpy()), 2)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 60.0)
        translation = wp.vec3(-0.1, 0, 0) - wp.quat_rotate(rotation, wp.vec3(-0.1, 0, -0.1))
        transform = np.array([*translation, *rotation], dtype=np.float32)
        s1.joint_q.assign(transform)
        newton.eval_fk(model, s1.joint_q, s1.joint_qd, s1)
        pipeline.collide(s1, contacts)
        solver.step(s1, s0, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        active = patches.view.weight.numpy() > 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_allclose(patches.current.anchor_a.numpy()[active, 0], -0.1, atol=1.0e-6)

    def test_shape_translation_retires_anchors_outside_new_geometry(self):
        """Retire the old contact footprint after changing a shape's local transform."""
        device = _DEVICE
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", friction_anchor_beta=0.2)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        self.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)

        transforms = model.shape_transform.numpy()
        transforms[box, 0] = 1.0
        model.shape_transform.assign(transforms)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        pipeline.collide(s1, contacts)
        patches.build(model, s1, contacts, shape_material_mu=model.shape_material_mu)
        active = patches.current.valid.numpy() != 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_array_equal(patches.current.source.numpy()[active], [-1, -1])
        local_x = patches.current.anchor_a.numpy()[active, 0]
        self.assertTrue(np.all(local_x >= 0.89), "friction still acts on the removed x=-0.1..0.1 footprint")

    def test_rejected_contact_history_keeps_correct_reset_world(self):
        """Keep world ownership valid when contact order changes and normal gates reject rows."""
        device = _DEVICE
        template = newton.ModelBuilder()
        template.add_ground_plane()
        body = template.add_body()
        template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverSteel(
            model, solve_mode="matrix_free", friction_anchor_beta=0.2, contact_gap_gate=0.001
        )
        contacts = newton.CollisionPipeline(model, rigid_contact_max=2).contacts()
        shape_body = model.shape_body.numpy()
        boxes = np.flatnonzero(shape_body >= 0)
        grounds = np.flatnonzero(shape_body < 0)
        contacts.rigid_contact_count.assign([2])
        contacts.rigid_contact_shape0.assign(boxes)
        contacts.rigid_contact_shape1.assign(grounds)
        contacts.rigid_contact_normal.assign([[0, 0, -1]] * 2)
        contacts.rigid_contact_point0.zero_()
        contacts.rigid_contact_point1.zero_()
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        self.assertEqual(int(patches.previous.valid.numpy().sum()), 2)

        contacts.rigid_contact_shape0.assign(boxes[::-1].copy())
        contacts.rigid_contact_shape1.assign(grounds[::-1].copy())
        contacts.rigid_contact_point0.assign([[0, 0, 0.002]] * 2)
        solver.step(s1, s0, model.control(), contacts, 0.005)
        self.assertTrue((solver._contact_slot.numpy()[:2] == -1).all())
        bodies = patches.previous.body_a.numpy()[:2]
        worlds = model.body_world.numpy()[bodies]
        np.testing.assert_array_equal(np.sort(worlds), [0, 1])
        solver.reset(s0, wp.array([True, False, False], dtype=bool, device=device))
        valid = patches.previous.valid.numpy()[:2]
        np.testing.assert_array_equal(valid, (worlds != 0).astype(np.int32))

    def test_shape_updates_refresh_geometry_and_keep_history(self):
        """Refresh broadphase bounds without discarding anchors on unchanged geometry."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=_DEVICE)
        solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", friction_anchor_beta=0.2)
        patches = solver._friction_patches
        radius = patches.body_radius.numpy().copy()
        patches.previous.valid.fill_(1)
        model.shape_collision_radius.assign(model.shape_collision_radius.numpy() * 2)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        np.testing.assert_allclose(patches.body_radius.numpy(), 2 * radius)
        self.assertEqual(np.count_nonzero(patches.previous.valid.numpy()), patches.previous.valid.shape[0])

    def test_gap_filtered_step_keeps_anchor_history(self):
        """Carry a region's anchors through a step in which the gap filter removes every friction row."""
        device = _DEVICE
        with wp.ScopedDevice(device):
            _model, solver, _contacts, step = _ground_box(device, contact_friction_gap_max=0.001)
            patches = solver._friction_patches
            for _ in range(40):
                step()
            self.assertGreater(int((patches.current.source.numpy() >= 0).sum()), 0)
            solver.contact_friction_gap_max = -1.0
            step()
            self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)
            self.assertGreater(int(patches.previous.valid.numpy().sum()), 0)
            solver.contact_friction_gap_max = 0.001
            step()
            weights = patches.view.weight.numpy()
            sources = patches.current.source.numpy()
            self.assertGreater(int(((weights > 0) & (sources >= 0)).sum()), 0)

    def test_notified_friction_edits_reach_the_patch_builder(self):
        """Build patches from notified friction edits, in place or replaced, in eager and captured steps."""
        for replace in (False, True):
            for capture in (False, True):
                with self.subTest(replace=replace, capture=capture), wp.ScopedDevice(_DEVICE):
                    _check_notified_friction_edit(self, _DEVICE, replace, capture)


@unittest.skipUnless(wp.is_cuda_available(), "SolverSteel requires CUDA")
class TestSteelFrictionAnchorSelection(unittest.TestCase):
    """Anchor selection, friction rows and reported contact forces of patch friction."""

    def test_anchor_selection_respects_contact_gap_filters(self):
        """Keep filtered extreme points from removing friction from a loaded middle contact."""
        for gate in ("contact_friction_gap_max", "contact_gap_gate"):
            with self.subTest(gate=gate), wp.ScopedDevice(_DEVICE):
                builder = newton.ModelBuilder()
                ground = builder.add_ground_plane()
                body = builder.add_body()
                box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize()
                solver = newton.solvers.SolverSteel(
                    model, solve_mode="matrix_free", friction_anchor_beta=0.2, **{gate: 0.001}
                )
                contacts = newton.CollisionPipeline(model, rigid_contact_max=3).contacts()
                contacts.rigid_contact_count.assign([3])
                contacts.rigid_contact_shape0.assign([box] * 3)
                contacts.rigid_contact_shape1.assign([ground] * 3)
                contacts.rigid_contact_point0.assign([[-0.08, 0, 0.002], [0, 0, 0], [0.08, 0, 0.002]])
                contacts.rigid_contact_point1.assign([[-0.08, 0, 0], [0, 0, 0], [0.08, 0, 0]])
                contacts.rigid_contact_normal.assign([[0, 0, -1]] * 3)
                solver.step(model.state(), model.state(), model.control(), contacts, 0.005)
                rows = solver._mf_row_type.numpy()[0, : solver._mf_constraint_count.numpy()[0]]
                self.assertEqual(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_FRICTION), 2)
                np.testing.assert_allclose(solver._friction_patches.view.weight.numpy()[:3], [0, 1, 0])

    def test_cuda_planar_patch_reduces_friction_rows(self):
        """Retain a box face's four normals while using only two friction anchors."""
        with wp.ScopedDevice(_DEVICE):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize()
            solver = newton.solvers.SolverSteel(
                model, solve_mode="matrix_free", friction_anchor_beta=0.2, articulation_constraint_max=64
            )
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="latest")
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            solver.step(state, model.state(), model.control(), contacts, 0.005)
            rows = np.concatenate(
                [
                    solver._row_type.numpy()[0, : solver._constraint_count.numpy()[0]],
                    solver._mf_row_type.numpy()[0, : solver._mf_constraint_count.numpy()[0]],
                ]
            )
            normals = int(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_CONTACT))
            friction = int(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_FRICTION))
            self.assertEqual(normals, 4)
            self.assertEqual(friction, 4)

    def test_contact_churn_reports_linear_force_but_not_a_wrench(self):
        """Compare the linear export with solved rows under an off-centre load after churn."""
        device = _DEVICE
        builder = newton.ModelBuilder()
        ground = builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", friction_anchor_beta=0.2, iterations=64)
        # A pipeline publishes the contact capacity; the contacts are written by hand and never collided.
        contacts = newton.CollisionPipeline(model, rigid_contact_max=4, soft_contact_max=0).contacts()
        contacts.rigid_contact_count.assign([4])
        contacts.rigid_contact_shape0.assign([box] * 4)
        contacts.rigid_contact_shape1.assign([ground] * 4)
        contacts.rigid_contact_normal.assign([[0, 0, -1]] * 4)
        points = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], dtype=np.float32)
        contacts.rigid_contact_point0.assign(points + np.array([0, 0, -0.1], dtype=np.float32))
        contacts.rigid_contact_point1.assign(points)
        s0, s1 = model.state(), model.state()
        dt = 0.005
        solver.step(s0, s1, model.control(), contacts, dt)
        points[:, :2] *= 0.8
        contacts.rigid_contact_point0.assign(points + np.array([0, 0, -0.1], dtype=np.float32))
        contacts.rigid_contact_point1.assign(points)
        # The x force and z torque correspond to an off-centre tangential load.
        s1.body_f.assign([[1, 0, 0, 0, 0, 0.05]])
        observables = solver.observables({newton.solvers.SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
        solver.step(s1, s0, model.control(), contacts, dt, observables=observables)
        self.assertGreater(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0), 0)
        count = int(solver._mf_constraint_count.numpy()[0])
        self.assertTrue(np.all(solver._contact_path.numpy()[:4] == 1))
        solved = np.einsum("ij,i->j", solver._mf_J_a.numpy()[0, :count], solver._mf_impulses.numpy()[0, :count]) / dt
        self.assertGreater(np.linalg.norm(solved[3:]), 0.01, "fixture must exercise a nonzero contact moment")
        reported = observables.contact_f.numpy().sum(axis=0)
        np.testing.assert_allclose(reported[:3], solved[:3], atol=1.0e-4)
        # Torque is explicitly unsupported by this export, including on the base
        # branch. Preserve that limitation visibly instead of asserting a wrench.
        np.testing.assert_array_equal(reported[3:], [0, 0, 0])

    def test_contact_forces_report_per_row_normal_load(self):
        """Report each contact's own normal force so a resting box's contact forces sum to its weight."""
        device = _DEVICE
        with wp.ScopedDevice(device):
            model, solver, contacts, step = _ground_box(device)
            observables = solver.observables({newton.solvers.SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            for _ in range(100):
                step(observables)
            count = int(contacts.rigid_contact_count.numpy()[0])
            forces = observables.contact_f.numpy()[:count, :3]
        weight = float(model.body_mass.numpy()[0]) * 9.81
        self.assertGreaterEqual(count, 3)
        self.assertAlmostEqual(float(abs(forces[:, 2].sum())), weight, delta=0.05 * weight)
        self.assertLess(float(np.abs(forces[:, 2]).max()), weight)


def _check_notified_friction_edit(test, device, replace, capture):
    """Edit friction after the anchors formed, notify, and require the next patch build to read it."""
    # A box held on a 0.3 rad incline by static friction carries patch anchors.
    builder = newton.ModelBuilder(gravity=(9.81 * np.sin(0.3), 0.0, -9.81 * np.cos(0.3)))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", iterations=32)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    def advance():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 0.005)
        state_0.assign(state_1)

    def patch_mu():
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        return solver._friction_patches.current.mu.numpy()[:count]

    for _ in range(20):
        advance()
    test.assertGreater(int(solver._friction_patches.previous.valid.numpy().sum()), 0)
    np.testing.assert_array_equal(patch_mu(), np.float32(0.5))
    if capture:
        with wp.ScopedCapture(device=device) as graph:
            advance()

    def run():
        if capture:
            wp.capture_launch(graph.graph)
        else:
            advance()

    mu = np.full(model.shape_count, 0.8, dtype=np.float32)
    if replace:
        model.shape_material_mu = wp.array(mu, dtype=wp.float32, device=device)
    else:
        model.shape_material_mu.assign(mu)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    run()
    np.testing.assert_array_equal(patch_mu(), np.float32(0.8))
    # Like the contact rows, the builder sees friction edits only through a notification.
    model.shape_material_mu.fill_(0.3)
    run()
    np.testing.assert_array_equal(patch_mu(), np.float32(0.8))


def test_disabled_material_options_keep_default_patches(test, device):
    """Keep ordinary patch defaults when the experimental material law is not enabled."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.05)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free")
    test.assertEqual(solver.friction_anchor_beta, 0.2)
    test.assertTrue(solver._friction_anchors_enabled)
    test.assertFalse(solver._contact_torsion_enabled)


def test_torsion_opt_in_keeps_default_patches(test, device):
    """Resolve omitted patch gain to default patch friction alongside torsion, without a warning."""
    options = {"device": device, "center_only": True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual, solver, *_ = fixture(0.01, friction_anchor_beta=None, **options)
    test.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
    reference, explicit, *_ = fixture(0.01, friction_anchor_beta=0.2, **options)
    test.assertTrue(solver._friction_anchors_enabled)
    test.assertTrue(explicit._friction_anchors_enabled)
    test.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
    test.assertGreater(solver._torsion_stats["rows"], 0)
    for name in reference:
        np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)


class TestPatchMaterialCompatibility(unittest.TestCase):
    pass


for _fn in (test_disabled_material_options_keep_default_patches, test_torsion_opt_in_keeps_default_patches):
    add_function_test(TestPatchMaterialCompatibility, _fn.__name__, _fn, devices=get_cuda_test_devices())


_COUPLED_MODES = MODES[1:]
_MODE_MU = 0.5
_MODE_SLIP = math.atan(_MODE_MU)
_MODE_TOLERANCE = 1.0e-4

# Each colored sweep variant: its kernel overrides and a tag of the kernel it must build.
_COLORED_VARIANTS = {
    "packed": ({"colored_packed": True}, "_pf1_st0_pk1"),
    "prefetch": ({"colored_packed": False}, "_pf1_st0_pk0"),
    "staged": ({"colored_staged": True}, "_pf1_st1"),
    "reference": ({"colored_prefetch": False}, "_pf0_st0_pk0_sl1"),
    "row loop": ({"colored_prefetch": False, "colored_straight_line": False}, "_sl0"),
    "block": ({}, "_sb"),
    "block global": ({}, "_gb"),
}


def _select_colored_variant(test, solver, variant):
    """Give ``solver`` the block sweeps of ``variant`` and check the kernel it runs."""
    if variant.startswith("block"):
        # More bodies than the staging cap keep the twists in global memory.
        bodies = 385 if variant == "block global" else solver._max_propagation_bodies
        solver._pgs_solve_propagation_colored_block_kernel = _get_pgs_solve_propagation_colored_block_kernel(
            solver._propagation_max_constraints,
            solver._color_world_offsets.shape[0] // solver._world_count,
            64,
            bodies,
            solver.model.device.arch,
            friction_mode=solver._friction_mode,
        )
        solver._pgs_solve_propagation_colored_warp_kernel = None
    warp_kernel = solver._pgs_solve_propagation_colored_warp_kernel
    kernel = solver._pgs_solve_propagation_colored_block_kernel if warp_kernel is None else warp_kernel
    test.assertIn(_COLORED_VARIANTS[variant][1], kernel.key)
    mode = solver._friction_mode
    test.assertEqual(kernel.key.endswith(mode), mode != "fixed_normal", kernel.key)


class _ModeScene:
    """One box on an incline or one sliding sphere per world, stepped in place so steps can be captured."""

    def __init__(
        self,
        test,
        device,
        response,
        mode,
        *,
        scene="incline",
        angle=0.0,
        articulated=False,
        worlds=1,
        variant=None,
        **solver_kwargs,
    ):
        world = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg = newton.ModelBuilder.ShapeConfig(mu=_MODE_MU, gap=0.0, restitution=0.0, density=500.0)
        tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle))
        world.add_shape_box(-1, hx=2.0, hy=2.0, hz=0.05, xform=wp.transform((0.0, 0.0, -0.05), tilt), cfg=cfg)
        if scene == "incline":
            pose = wp.transform(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 0.05)), tilt)
        else:
            pose = wp.transform((0.0, 0.0, 0.05), wp.quat_identity())
        body = (world.add_link if articulated else world.add_body)(xform=pose)
        if articulated:
            joint = world.add_joint_d6(
                parent=-1,
                child=body,
                parent_xform=pose,
                linear_axes=[newton.ModelBuilder.JointDofConfig(axis=axis) for axis in newton.Axis],
                angular_axes=[newton.ModelBuilder.JointDofConfig(axis=axis) for axis in newton.Axis],
            )
            world.add_articulation([joint])
        if scene == "incline":
            world.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05, cfg=cfg)
        else:
            world.add_shape_sphere(body, radius=0.05, cfg=cfg)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(world, worlds)
        self.model = builder.finalize(device=device)
        args = {
            "articulated_contact_response": response,
            "friction_mode": mode,
            "friction_anchor_beta": 0.0,
            "iterations": 32,
            "free_body_constraint_max": 64,
            "articulation_constraint_max": 64,
            "row_watermark": True,
            "deterministic": wp.DeterministicMode.RUN_TO_RUN,
        }
        args.update(solver_kwargs)
        overrides = _COLORED_VARIANTS[variant][0] if variant else {}
        with kernel_overrides(overrides):
            self.solver = SolverSteel(self.model, **args)
        if variant:
            _select_colored_variant(test, self.solver, variant)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=16 * worlds, deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.state_in, self.state_out = self.model.state(), self.model.state()
        self.control = self.model.control()
        if scene == "sphere":
            # Sliding along x without spin.
            qd = self.state_in.joint_qd.numpy().reshape(worlds, 6)
            qd[:, 0] = 2.0
            self.state_in.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(self.model, self.state_in.joint_q, self.state_in.joint_qd, self.state_in)

    def step(self, steps=1):
        for _ in range(steps):
            self.state_in.clear_forces()
            self.pipeline.collide(self.state_in, self.contacts)
            self.solver.step(self.state_in, self.state_out, self.control, self.contacts, DT)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                wp.copy(getattr(self.state_in, name), getattr(self.state_out, name))

    def velocity(self):
        return self.state_in.body_qd.numpy().astype(np.float64)

    def run(self, steps):
        trace = []
        for _ in range(steps):
            self.step()
            trace.append(self.velocity())
        return np.asarray(trace)


def _mode_trace(test, device, response, mode, steps, **kwargs):
    return _ModeScene(test, device, response, mode, **kwargs).run(steps)


def test_colored_incline_tracks_immediate(test, device):
    """Follow immediate's coupled update of each mode on a box near the slip angle, which fixed_normal does not.

    The mode's gap to fixed_normal on immediate must exceed ten times the tolerance, so colored solving the box
    with fixed_normal would miss by that much.
    """
    for angle, iterations in itertools.product((_MODE_SLIP - 0.05, _MODE_SLIP + 0.05), (4, 32)):
        fixed = _mode_trace(test, device, "immediate", "fixed_normal", 20, angle=angle, iterations=iterations)
        for mode in _COUPLED_MODES:
            with test.subTest(angle=angle, iterations=iterations, mode=mode):
                immediate = _mode_trace(test, device, "immediate", mode, 20, angle=angle, iterations=iterations)
                scene = _ModeScene(test, device, "propagation_colored", mode, angle=angle, iterations=iterations)
                colored = scene.run(20)
                for key, value in scene.solver.constraint_row_watermarks().items():
                    if "overflow" in key or "dropped" in key:
                        test.assertEqual(value, 0, key)
                np.testing.assert_allclose(colored, immediate, atol=_MODE_TOLERANCE, rtol=0.0)
                # coulomb_newton converges to the fixed_normal solution.
                if iterations == 4 or mode != "coulomb_newton":
                    test.assertGreater(np.abs(immediate - fixed).max(), 10.0 * _MODE_TOLERANCE)


def test_colored_articulated_contacts_take_the_mode(test, device):
    """Slide a box on a six-DOF joint as the free box does, in every mode.

    Colored solves every contact with the coupled update. Immediate's articulated contacts run bisection
    without the de Saxce bias and fixed_normal for coulomb_newton, so it matches only in bisection. The de
    Saxce bias lifts the sliding box off the incline at step 7, which grows the joint's float32 rounding gap
    past the tolerance, so that mode is compared for 6 steps.
    """
    for angle, iterations in itertools.product((_MODE_SLIP - 0.05, _MODE_SLIP + 0.05), (4, 32)):
        for mode in _COUPLED_MODES:
            with test.subTest(angle=angle, iterations=iterations, mode=mode):
                steps = 6 if mode == "bisection_desaxce" else 12
                kwargs = {"angle": angle, "iterations": iterations}
                free = _mode_trace(test, device, "propagation_colored", mode, steps, **kwargs)
                articulated = _mode_trace(test, device, "propagation_colored", mode, steps, articulated=True, **kwargs)
                np.testing.assert_allclose(articulated, free, atol=_MODE_TOLERANCE, rtol=0.0)
                if mode == "bisection":
                    immediate = _mode_trace(test, device, "immediate", mode, steps, articulated=True, **kwargs)
                    np.testing.assert_allclose(articulated, immediate, atol=_MODE_TOLERANCE, rtol=0.0)


def test_colored_sliding_sphere_rolls_as_on_immediate(test, device):
    """Slide a sphere launched without spin until it rolls at 5/7 of its speed, as on immediate."""
    for mode in _COUPLED_MODES:
        with test.subTest(mode=mode):
            immediate = _mode_trace(test, device, "immediate", mode, 120, scene="sphere")
            colored = _mode_trace(test, device, "propagation_colored", mode, 120, scene="sphere")
            np.testing.assert_allclose(colored, immediate, atol=_MODE_TOLERANCE, rtol=0.0)
            speed, spin = colored[-1, 0, 0], colored[-1, 0, 4]
            test.assertAlmostEqual(speed, 2.0 * 5.0 / 7.0, delta=0.02)
            test.assertAlmostEqual(spin * 0.05, speed, delta=1.0e-3)


def test_colored_kinematic_support_drags_the_box(test, device):
    """Carry a free and an articulated box on a moving kinematic support at its speed in every mode.

    The colored sweep folds the support's velocity into the sliding rows' rhs, which the coupled update reads.
    """
    for mode, articulated in itertools.product(_COUPLED_MODES, (False, True)):
        with test.subTest(mode=mode, articulated=articulated):
            mean_vx, *_ = _box_on_support(device, mode, articulated, 0.5, response="propagation_colored")
            test.assertAlmostEqual(mean_vx, 0.5, delta=5.0e-3)


def test_every_colored_variant_agrees(test, device):
    """Run the coupled update in every colored sweep, agreeing within float32 summation order.

    Every sweep calls the same coupled update for each mode, so one mode covers them.
    """
    mode = "bisection_desaxce"
    reference = None
    for variant in _COLORED_VARIANTS:
        with test.subTest(variant=variant):
            if variant == "staged" and not _colored_staging_supported(device.arch, 1, 2):
                continue
            kwargs = {"angle": _MODE_SLIP + 0.05, "iterations": 4, "variant": variant}
            # The staged sweep packs one world per block.
            trace = _mode_trace(test, device, "propagation_colored", mode, 20, worlds=2, **kwargs)
            if reference is None:
                reference = trace
            # A bisection or a slip flip turns float32 summation-order differences into larger steps.
            np.testing.assert_allclose(trace[0], reference[0], atol=1.0e-5, rtol=0.0)
            np.testing.assert_allclose(trace, reference, atol=_MODE_TOLERANCE, rtol=0.0)


def test_colored_modes_repeat_bitwise_and_replay(test, device):
    """Repeat several deterministic worlds bitwise, and replay a captured step as eager steps, in every mode."""
    for mode in _COUPLED_MODES:
        with test.subTest(mode=mode):
            kwargs = {"angle": _MODE_SLIP + 0.05, "worlds": 3}
            runs = [_mode_trace(test, device, "propagation_colored", mode, 30, **kwargs) for _ in range(2)]
            np.testing.assert_array_equal(runs[0], runs[1])
            eager = _ModeScene(test, device, "propagation_colored", mode, **kwargs)
            eager.step(2)
            expected = eager.run(10)
            captured = _ModeScene(test, device, "propagation_colored", mode, **kwargs)
            captured.step(2)
            with wp.ScopedCapture(device=device) as capture:
                captured.step()
            trace = []
            for _ in range(10):
                wp.capture_launch(capture.graph)
                trace.append(captured.velocity())
            np.testing.assert_array_equal(np.asarray(trace), expected)


def test_colored_modes_keep_the_coupled_restrictions(test, device):
    """Keep the coupled modes' restrictions on colored, select point friction by default, and reject plain propagation."""
    model = _ModeScene(test, device, "immediate", "fixed_normal").model
    for mode in _COUPLED_MODES:
        with test.subTest(mode=mode):
            colored = {"articulated_contact_response": "propagation_colored", "friction_mode": mode}
            with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "point friction"):
                solver = SolverSteel(model, **colored)
            test.assertEqual(solver.friction_anchor_beta, 0.0)
            test.assertIn(f"_{mode}", solver._pgs_solve_propagation_colored_warp_kernel.key)
            with test.assertRaisesRegex(ValueError, "friction_mode='fixed_normal'"):
                SolverSteel(model, **colored, friction_anchor_beta=0.2)
            with test.assertRaisesRegex(ValueError, "friction_mode"):
                SolverSteel(model, **colored, friction_anchor_beta=0.0, contact_compliance=True)
            with test.assertRaisesRegex(NotImplementedError, "friction_mode"):
                SolverSteel(
                    model, articulated_contact_response="propagation", friction_mode=mode, friction_anchor_beta=0.0
                )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = SolverSteel(model, articulated_contact_response="propagation_colored")
    test.assertFalse(default._pgs_solve_propagation_colored_warp_kernel.key.endswith(_COUPLED_MODES))


class TestSteelColoredFrictionMode(unittest.TestCase):
    pass


for _fn in (
    test_colored_incline_tracks_immediate,
    test_colored_articulated_contacts_take_the_mode,
    test_colored_sliding_sphere_rolls_as_on_immediate,
    test_colored_kinematic_support_drags_the_box,
    test_every_colored_variant_agrees,
    test_colored_modes_repeat_bitwise_and_replay,
    test_colored_modes_keep_the_coupled_restrictions,
):
    add_function_test(TestSteelColoredFrictionMode, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
