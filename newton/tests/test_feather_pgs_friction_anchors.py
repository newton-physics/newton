"""Tests for FeatherPGS positional friction anchors (``friction_anchor_beta``)."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    compute_mf_effective_mass_and_rhs,
    compute_mf_rhs_bias,
    compute_propagation_effective_mass_and_rhs,
    compute_propagation_rhs_bias,
    compute_world_contact_bias,
)

_MU_JAW, _MU_BOX = 5.0, 0.5
_BOX_HALF, _JAW_HALF_T, _GAP0 = 0.02, 0.005, 0.001


def _build_v_jaws(tilt_deg: float, geometry: str = "box"):
    """Fixed base, two prismatic jaws (left driven, right slaved through a mimic row) pinching a
    free 0.1 kg box. Both jaw faces tilt toward +z by ``tilt_deg`` so the two normal rows'
    depenetration biases share a +z tangential component: the geometry that leaks tangential
    drift through velocity-only friction rows (a 2F-85 pad pair has ~4 deg of such tilt)."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.rigid_gap = 0.005
    base = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="base")
    b.add_shape_box(
        base,
        hx=0.05,
        hy=0.05,
        hz=0.005,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, has_shape_collision=False),
    )
    root = b.add_joint_fixed(-1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    jaw_cfg = newton.ModelBuilder.ShapeConfig(density=2000.0, mu=_MU_JAW)
    x0 = _BOX_HALF + _JAW_HALF_T + _GAP0
    jaws, joints = [], []
    for name, sign in (("jaw_L", -1.0), ("jaw_R", 1.0)):
        body = b.add_link(xform=wp.transform(wp.vec3(sign * x0, 0.0, 0.5), wp.quat_identity()), label=name)
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-sign * np.radians(tilt_deg)))
        b.add_shape_box(body, hx=_JAW_HALF_T, hy=0.02, hz=0.02, xform=wp.transform(wp.vec3(0.0), rot), cfg=jaw_cfg)
        driven = sign < 0
        joints.append(
            b.add_joint_prismatic(
                base,
                body,
                parent_xform=wp.transform(wp.vec3(sign * x0, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.X,
                limit_lower=-0.05,
                limit_upper=0.05,
                target_pos=0.01 if driven else 0.0,
                target_ke=100.0 if driven else 0.0,
                target_kd=10.0 if driven else 0.0,
            )
        )
        jaws.append(body)
    b.add_articulation([root, *joints], label="gripper")
    b.set_joint_mimic(joints[1], joints[0], coeffs=(0.0, -1.0))
    for dof in range(len(b.joint_effort_limit)):
        b.joint_effort_limit[dof] = 10.0
    box = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="box")
    if geometry == "sphere":
        b.add_shape_sphere(
            box,
            radius=_BOX_HALF,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.1 / (4.0 / 3.0 * np.pi * _BOX_HALF**3), mu=_MU_BOX),
        )
    elif geometry == "capsule":
        b.add_shape_capsule(
            box,
            radius=_BOX_HALF,
            half_height=_BOX_HALF,
            xform=wp.transform(wp.vec3(0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.1 / (np.pi * _BOX_HALF**2 * 2 * _BOX_HALF + 4.0 / 3.0 * np.pi * _BOX_HALF**3), mu=_MU_BOX
            ),
        )
    else:
        b.add_shape_box(
            box,
            hx=_BOX_HALF,
            hy=_BOX_HALF,
            hz=_BOX_HALF,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.1 / (2 * _BOX_HALF) ** 3, mu=_MU_BOX),
        )
    return b.finalize(), jaws, box


_SQUEEZE_SOLVER = {
    "pgs_mode": "matrix_free",
    "articulated_contact_response": "immediate",
    "pgs_iterations": 64,
    "pgs_velocity_iterations": 0,
    "pgs_beta": 0.05,
    "pgs_contact_regularization": 0.01,
    "contact_friction_gap_threshold": 0.001,
    "contact_friction_position_iterations": 4,
    "contact_shared_anchor": True,
    "contact_friction_shared_anchor": True,
    "enable_bilateral_preelimination": True,
}


def _run_squeeze(
    tilt_deg: float, steps: int, dt: float = 0.005, matching: str = "latest", substeps: int = 1, **solver_kwargs
):
    """Return the box's z drift relative to the jaws (m, positive = up) and the solver."""
    model, jaws, box = _build_v_jaws(tilt_deg)
    kwargs = dict(_SQUEEZE_SOLVER)
    kwargs.update(solver_kwargs)
    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn", contact_matching=matching)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    rel = []
    for _ in range(steps):
        pipeline.collide(s0, contacts)  # once per environment step; substeps reuse the buffer
        for _sub in range(substeps):
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt / substeps)
            s0, s1 = s1, s0
        bq = s0.body_q.numpy()
        rel.append(bq[box][2] - bq[jaws[0]][2])
    rel = np.asarray(rel)
    settle = int(0.5 / dt)  # skip the free fall before the pinch closes
    return rel[-1] - rel[settle], solver, s0


def _build_incline(theta_deg: float, mu: float):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.rigid_gap = 0.005
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.radians(theta_deg)))
    # static slab as the incline (shape on the world body)
    b.add_shape_box(
        -1,
        hx=2.0,
        hy=0.5,
        hz=0.05,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), rot),
        cfg=newton.ModelBuilder.ShapeConfig(mu=mu),
    )
    n = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))  # slab normal
    box = b.add_body(xform=wp.transform(n * (0.05 + 0.0002), rot), label="box")
    b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu))
    model = b.finalize()
    return model, box, rot


def _run_incline(steps: int, dt: float, **solver_kwargs):
    theta, mu = 30.0, 0.3
    model, box, _rot = _build_incline(theta, mu)
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=32, pgs_beta=0.05, **solver_kwargs
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, broad_phase="nxn", contact_matching="latest")
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    v = s0.body_qd.numpy()[box][:3]  # Newton body_qd = [linear, angular]
    down_slope = np.array([np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))])  # slab tilted about +y by theta
    return float(np.dot(v, down_slope)), float(np.linalg.norm(v - np.dot(v, down_slope) * down_slope)), solver, contacts


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSFrictionAnchors(unittest.TestCase):
    def setUp(self):
        # The shared-anchor squeeze fixture intentionally combines contact_shared_anchor with patch friction.
        filters = warnings.catch_warnings()
        filters.__enter__()
        self.addCleanup(filters.__exit__, None, None, None)
        warnings.filterwarnings(
            "ignore", message=r"Patch friction selects its own friction locations", category=UserWarning
        )
        # The per-route row-builder check runs propagation routes with the default pre-elimination.
        warnings.filterwarnings(
            "ignore",
            message=r"SolverFeatherPGS: bilateral pre-elimination does not support propagation",
            category=UserWarning,
        )

    def test_explicit_opt_out_keeps_friction_rows_velocity_only(self):
        """Preserve velocity-only rows and omit anchor state with ``friction_anchor_beta=0``."""
        model, _jaws, _box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverFeatherPGS(model, **_SQUEEZE_SOLVER, friction_anchor_beta=0.0)
        self.assertFalse(solver._friction_anchors_enabled)
        self.assertFalse(hasattr(solver._friction_patches, "current"))
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn")  # no matching needed
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
        row_type = solver.row_type.numpy()
        friction = row_type == PGS_CONSTRAINT_TYPE_FRICTION
        self.assertGreater(int(friction.sum()), 0)
        self.assertEqual(float(np.abs(solver.phi.numpy()[friction]).max()), 0.0)
        self.assertEqual(float(np.abs(solver.row_beta.numpy()[friction]).max()), 0.0)

    def test_anchors_do_not_require_contact_matching(self):
        """Keep patch history owned by the solver, independently of collision matching."""
        model, _jaws, _box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverFeatherPGS(model, **_SQUEEZE_SOLVER, friction_anchor_beta=0.2)
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
            abs(drift_off), 1.5e-4, f"tilted pinch should leak drift without anchors, got {drift_off:.2e} m"
        )
        self.assertLess(
            abs(drift_on),
            0.3 * abs(drift_off),
            f"anchored pinch (beta 0.05) drifted {drift_on:.2e} m vs {drift_off:.2e} m",
        )
        self.assertLess(abs(drift_on_02), 5.0e-5, f"anchored pinch (beta 0.2) drifted {drift_on_02:.2e} m")
        # the held contacts are anchored (not sliding) at the end of the hold
        valid = solver_on._friction_patches.current.valid.numpy()
        self.assertGreater(int(valid.sum()), 0)
        flat_off, _, _ = _run_squeeze(0.0, steps, friction_anchor_beta=0.0)
        # sticky matching replays body-local witness points; anchors must still hold
        drift_sticky, _, _ = _run_squeeze(5.0, steps, matching="sticky", friction_anchor_beta=0.2)
        self.assertLess(abs(drift_sticky), 1.0e-4, f"anchored pinch (sticky matching) drifted {drift_sticky:.2e} m")
        # two solver substeps on one collide per environment step (Isaac Lab's loop): anchors
        # must map 1:1 on the reused buffer, not through the stale match index
        drift_sub, _, _ = _run_squeeze(5.0, steps, substeps=2, friction_anchor_beta=0.2)
        self.assertLess(abs(drift_sub), 1.0e-4, f"anchored pinch with 2 substeps per collide drifted {drift_sub:.2e} m")
        flat_on, _, _ = _run_squeeze(0.0, steps, friction_anchor_beta=0.05)
        self.assertLess(abs(flat_on), 1.0e-4)
        self.assertLess(abs(flat_off), 1.0e-4)

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
        # every loaded contact slid this step, so its anchor was dropped
        n = int(contacts.rigid_contact_count.numpy()[0])
        if n > 0:
            self.assertEqual(int(solver._friction_patches.current.valid.numpy()[:n].sum()), 0)

    def test_row_builders_store_anchor_separation_per_route(self):
        """Store raw separation in dense rows and gain-scaled separation in the other routes."""
        beta = 0.3

        def check(model, solver_kwargs, matching="latest", steps=120):
            solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=beta, **solver_kwargs)
            pipeline = newton.CollisionPipeline(
                model, rigid_contact_max=256, broad_phase="nxn", contact_matching=matching
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
            slot = solver.contact_slot.numpy()[:n]
            path = solver.contact_path.numpy()[:n]
            checked = {0: 0, 1: 0, 2: 0}
            for c in range(n):
                if slot[c] < 0 or path[c] < 0:
                    continue
                if path[c] == 0:
                    rt, ph, rb = solver.row_type.numpy()[0], solver.phi.numpy()[0], solver.row_beta.numpy()[0]
                    if rt[slot[c] + 1] != PGS_CONSTRAINT_TYPE_FRICTION:
                        continue
                    np.testing.assert_allclose(ph[slot[c] + 1 : slot[c] + 3], fa[c], rtol=1.0e-6, atol=1.0e-9)
                    np.testing.assert_allclose(rb[slot[c] + 1 : slot[c] + 3], beta)
                else:
                    ph = (solver.mf_phi if path[c] == 1 else solver.propagation_phi).numpy()[0]
                    rt = (solver.mf_row_type if path[c] == 1 else solver.propagation_row_type).numpy()[0]
                    if rt[slot[c] + 1] != PGS_CONSTRAINT_TYPE_FRICTION:
                        continue
                    np.testing.assert_allclose(ph[slot[c] + 1 : slot[c] + 3], beta * fa[c], rtol=1.0e-5, atol=1.0e-9)
                checked[int(path[c])] += 1
            return checked

        # dense route: articulated jaws vs free box under immediate response
        model, _, _ = _build_v_jaws(5.0)
        c_dense = check(model, _SQUEEZE_SOLVER)
        self.assertGreater(c_dense[0], 0)
        # propagation route: same scene, contacts routed to propagation rows
        model, _, _ = _build_v_jaws(5.0)
        prop_kwargs = dict(_SQUEEZE_SOLVER, articulated_contact_response="propagation", pgs_contact_regularization=0.0)
        c_prop = check(model, prop_kwargs)
        self.assertGreater(c_prop[2], 0)
        # matrix-free route: free box resting on a static slab
        model, _, _ = _build_incline(0.0, 0.5)
        c_mf = check(model, {"pgs_mode": "matrix_free", "pgs_iterations": 16, "pgs_beta": 0.05})
        self.assertGreater(c_mf[1], 0)

    def test_graph_capture_replays(self):
        """Capture and replay two steps with device-side anchor history under a CUDA graph."""
        model, jaws, box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.05, **_SQUEEZE_SOLVER)
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
            solver.seed_double_buffer_events()
            step()
            step()
        for _ in range(200):
            wp.capture_launch(capture.graph)
        bq = s0.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(bq)))
        self.assertLess(abs(bq[box][2] - bq[jaws[0]][2]), 0.01)


def _rhs_for_family(family: str, *, phi, row_beta, pgs_beta, dt, bias_scale, device="cpu"):
    """RHS of one world with rows [CONTACT, FRICTION, FRICTION] for one RHS kernel family.

    ``phi`` holds what the row builders store: raw separation for dense rows (paired with
    ``row_beta``), gain-premultiplied separation for matrix-free and propagation rows.
    """
    contact, friction = PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
    count = wp.array([3], dtype=wp.int32, device=device)
    row_type = wp.array([[contact, friction, friction]], dtype=wp.int32, device=device)
    phi_arr = wp.array([list(phi)], dtype=wp.float32, device=device)
    zeros3 = lambda: wp.zeros((1, 3), dtype=wp.float32, device=device)  # noqa: E731
    neg1 = wp.full((1, 3), -1, dtype=wp.int32, device=device)
    zJ = lambda: wp.zeros((1, 3, 6), dtype=wp.float32, device=device)  # noqa: E731
    inf = wp.array([float("inf")], dtype=wp.float32, device=device)
    rhs = zeros3()
    if family == "dense":
        wp.launch(
            compute_world_contact_bias,
            dim=1,
            inputs=[
                count,
                3,
                phi_arr,
                wp.array([list(row_beta)], dtype=wp.float32, device=device),
                row_type,
                zeros3(),
                dt,
                bias_scale,
                1.0,
                1.0,
                1.0,
            ],
            outputs=[rhs, zeros3()],
            device=device,
        )
    elif family == "mf_setup":
        wp.launch(
            compute_mf_effective_mass_and_rhs,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                zJ(),
                zJ(),
                wp.zeros((1,), dtype=wp.spatial_matrix, device=device),
                phi_arr,
                row_type,
                zeros3(),
                zeros3(),
                0,
                wp.array([-1], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                inf,
                1.0e-6,
                pgs_beta,
                1.0,
                dt,
                1.0,
                0.5,
                3,
            ],
            outputs=[zeros3(), zJ(), zJ(), rhs, zeros3()],
            device=device,
        )
    elif family == "mf_velocity":
        wp.launch(
            compute_mf_rhs_bias,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                neg1,
                neg1,
                zJ(),
                zJ(),
                wp.zeros((1, 1), dtype=wp.int32, device=device),
                phi_arr,
                row_type,
                zeros3(),
                zeros3(),
                0,
                inf,
                pgs_beta,
                dt,
                bias_scale,
                1.0,
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                0,
                0,
                0.5,
                3,
            ],
            outputs=[rhs],
            device=device,
        )
    elif family == "propagation_setup":
        wp.launch(
            compute_propagation_effective_mass_and_rhs,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                zJ(),
                zJ(),
                wp.zeros((1, 6, 6), dtype=wp.float32, device=device),
                phi_arr,
                row_type,
                zeros3(),
                wp.zeros((1, 6), dtype=wp.float32, device=device),
                inf,
                1.0e-6,
                pgs_beta,
                1.0,
                dt,
                1.0,
                0.5,
                3,
            ],
            outputs=[zeros3(), zJ(), zJ(), rhs, zeros3(), zeros3()],
            device=device,
        )
    elif family == "propagation_velocity":
        wp.launch(
            compute_propagation_rhs_bias,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                zJ(),
                zJ(),
                phi_arr,
                row_type,
                zeros3(),
                inf,
                pgs_beta,
                dt,
                bias_scale,
                1.0,
                wp.zeros((1, 6), dtype=wp.float32, device=device),
                0,
                0,
                3,
            ],
            outputs=[rhs],
            device=device,
        )
    else:
        raise ValueError(family)
    return rhs.numpy()[0]


def _build_two_world_free_model(device):
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_free(parent=-1, child=body)
    template.add_articulation([joint])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    return builder.finalize(device=device)


class TestFeatherPGSFrictionAnchorKernels(unittest.TestCase):
    """Device-agnostic checks of the anchor bookkeeping (run on CPU)."""

    def test_rhs_bias_matches_on_every_row_family_and_vanishes_in_velocity_pass(self):
        """Apply equal positional bias on all row families and drop it in velocity-only passes."""
        pgs_beta, fa_beta, dt = 0.05, 0.3, 0.005
        phi_n, e0, e1 = -2.0e-3, 1.0e-4, -2.5e-4
        expect_pos = np.array([pgs_beta * phi_n / dt, fa_beta * e0 / dt, fa_beta * e1 / dt], dtype=np.float32)
        dense = lambda s: _rhs_for_family(  # noqa: E731
            "dense", phi=[phi_n, e0, e1], row_beta=[pgs_beta, fa_beta, fa_beta], pgs_beta=pgs_beta, dt=dt, bias_scale=s
        )
        pre = [phi_n, fa_beta * e0, fa_beta * e1]
        np.testing.assert_allclose(dense(1.0), expect_pos, rtol=1.0e-5)
        np.testing.assert_allclose(dense(0.0), 0.0, atol=1.0e-9)
        for family in ("mf_setup", "propagation_setup"):
            got = _rhs_for_family(family, phi=pre, row_beta=None, pgs_beta=pgs_beta, dt=dt, bias_scale=1.0)
            np.testing.assert_allclose(got, expect_pos, rtol=1.0e-5, err_msg=family)
        for family in ("mf_velocity", "propagation_velocity"):
            got = _rhs_for_family(family, phi=pre, row_beta=None, pgs_beta=pgs_beta, dt=dt, bias_scale=1.0)
            np.testing.assert_allclose(got, expect_pos, rtol=1.0e-5, err_msg=family)
            got0 = _rhs_for_family(family, phi=pre, row_beta=None, pgs_beta=pgs_beta, dt=dt, bias_scale=0.0)
            np.testing.assert_allclose(got0, 0.0, atol=1.0e-9, err_msg=family)

    def test_reset_clears_anchor_history_full_and_masked(self):
        """Drop carried anchors of selected worlds on reset, including without warm starting."""
        device = "cpu"
        model = _build_two_world_free_model(device)
        solver = newton.solvers.SolverFeatherPGS(
            model, pgs_mode="split", friction_anchor_beta=0.2, dense_max_constraints=4, mf_max_constraints=4
        )
        self.assertFalse(solver.pgs_warmstart)
        n = solver._friction_patches.previous.valid.shape[0]
        worlds = np.arange(n, dtype=np.int32) % 2
        for mask, expect_cleared in (
            (None, (True, True)),
            ((True, False), (True, False)),
            ((False, True), (False, True)),
        ):
            solver._friction_patches.previous.valid.fill_(1)
            solver._friction_patches.previous_world.assign(worlds)
            wm = None if mask is None else wp.array(mask, dtype=wp.bool, device=device)
            solver.reset(model.state(), wm)
            valid = solver._friction_patches.previous.valid.numpy()
            for world, cleared in enumerate(expect_cleared):
                sel = valid[worlds == world]
                np.testing.assert_array_equal(sel, 0 if cleared else 1, err_msg=f"mask={mask} world={world}")


if __name__ == "__main__":
    unittest.main()
