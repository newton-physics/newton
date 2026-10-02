# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the velocity-only iterations and speculative contacts of SolverFeatherPGS.

The final velocity pass rebuilds the contact RHS with the speculative term
scaled to zero. That is correct for a row whose position-solve trajectory
reaches the surface. For a row that remains separated inside the collision
margin, it rewrites the constraint from "you may close by the remaining gap"
(``u + phi/h >= 0``) into "you may not approach at all" (``u >= 0``), stopping
a falling body at the edge of the margin. The velocity pass distinguishes the
two using the position solution's linearized end gap, not impulse magnitude.

No restitution is involved anywhere here: this is ordinary free fall with
``pgs_velocity_iterations`` enabled.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import compute_mf_velocity_rhs
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

SIM_DT = 1.0 / 2000.0
RADIUS = 0.05
DROP = 0.12

# Values written into ``SolverFeatherPGS.contact_path``.
PATH_DENSE = 0
PATH_MATRIX_FREE = 1

# The route each scene takes, measured rather than assumed: a lone free body uses the
# free-body rows, and a contact on an articulated root uses the dense rows.
EXPECTED_PATHS = {
    "free": {PATH_MATRIX_FREE},
    "articulated": {PATH_DENSE},
}


def _make_solver(model, **kwargs):
    """Create the solver without angular damping, which would mask the contact response."""
    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
    solver.rigid_body_angular_damping.zero_()
    return solver


def _build(device, scene, mass=1.0):
    """Free sphere, or a two-link articulation whose root carries the contact."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    if scene == "free":
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    else:
        # add_link + explicit joints + add_articulation: add_body would create a
        # standalone free articulation per call, so appending a revolute to two
        # of them yields two articulations and an unowned loop joint rather than
        # the single two-link chain this fixture is meant to exercise.
        body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        link = builder.add_link(xform=wp.transform(wp.vec3(0.12, 0.0, RADIUS + DROP), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(link, radius=RADIUS, cfg=cfg)
        root_joint = builder.add_joint_free(child=body)
        hinge = builder.add_joint_revolute(
            parent=body,
            child=link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([root_joint, hinge])
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device), body


def _drop(device, velocity_iterations, steps=620, scene="free", *, pgs_warmstart=False):
    """Drop a body onto a plane; return heights, velocities, contact counts, routes."""
    model, body = _build(device, scene)
    solver = _make_solver(model, pgs_velocity_iterations=velocity_iterations, pgs_warmstart=pgs_warmstart)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    heights, velocities, counts, routes = [], [], [], set()
    pipeline = (
        newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        if pgs_warmstart
        else newton.CollisionPipeline(model)
    )
    contacts = pipeline.contacts()
    for _ in range(steps):
        contacts.clear()
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, SIM_DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy()[body][2]) - RADIUS)
        velocities.append(float(state_0.body_qd.numpy()[body][2]))
        n = int(contacts.rigid_contact_count.numpy()[0])
        counts.append(n)
        if n:
            routes.update(int(v) for v in solver.contact_path.numpy()[:n] if v >= 0)
    return np.asarray(heights), np.asarray(velocities), np.asarray(counts), routes


def _assert_landed(test, heights, velocities, counts, label):
    """Assert the body came to rest ON the plane, not through it and not above it."""
    final_h, final_v = float(heights[-1]), float(velocities[-1])
    # Bounded on BOTH sides: a tunnelling body has negative heights and would
    # satisfy any upper bound by itself.
    test.assertLess(abs(final_h), 0.005, f"{label}: rested at {final_h:+.4f} m instead of on the surface")
    test.assertLess(abs(final_v), 0.02, f"{label}: still moving at {final_v:+.4f} m/s")
    test.assertGreater(float(heights[:80].min()), 0.05, f"{label}: body was not still falling early in the run")
    test.assertGreater(int(counts.max()), 0, f"{label}: no contact was ever generated")
    settled = heights[-150:]
    test.assertLess(float(settled.max() - settled.min()), 0.002, f"{label}: resting height was not stable")


def _single_step(device, gap, approach_speed, velocity_iterations=4, mass=1.0):
    """One step of a body ``gap`` above the plane closing at ``approach_speed``.

    Returns the velocity before and after, the contact count, the position-pass
    impulse, the start gap, and the linearized end gap on positive-gap rows, so
    a test can confirm which side of the velocity-pass classifier it exercised.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))  # no gravity: isolate the constraint
    # density=0 so the body's mass is exactly the requested value: the point of
    # the light-body case is that the position impulse scales with it.
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + gap), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(np.eye(3) * 0.4 * mass * RADIUS**2),
    )
    builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    # The light-body case sits below the inertia floor on purpose.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Inertia validation corrected", category=UserWarning)
        model = builder.finalize(device=device)
    solver = _make_solver(model, pgs_velocity_iterations=velocity_iterations)
    # The position solve does not depend on the velocity-only iterations, so a twin
    # solver without them exposes the position-solve rows, impulses and velocity.
    position_solver = _make_solver(model)
    state_0, state_1 = model.state(), model.state()
    # A free body is driven through joint_qd; writing body_qd alone is discarded
    # by the forward kinematics the solver runs from joint state.
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[2] = -abs(approach_speed)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    before = float(state_0.body_qd.numpy()[body][2])
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    state_0.clear_forces()
    position_state = model.state()
    position_solver.step(state_0, position_state, model.control(), contacts, SIM_DT)
    solver.step(state_0, state_1, model.control(), contacts, SIM_DT)
    n = int(contacts.rigid_contact_count.numpy()[0])
    rows = int(position_solver.mf_constraint_count.numpy()[0])
    phi = position_solver.mf_phi.numpy()[0][:rows]
    lam = position_solver.mf_impulses.numpy()[0][:rows]
    row_type = position_solver.mf_row_type.numpy()[0][:rows]
    contact_rows = np.flatnonzero((row_type == 0) & (phi > 0.0))
    position_impulse = float(lam[contact_rows].max()) if contact_rows.size else 0.0
    max_phi = float(phi[contact_rows].max()) if contact_rows.size else 0.0
    v_position = position_solver.v_out.numpy()
    J_a = position_solver.mf_J_a.numpy()[0]
    J_b = position_solver.mf_J_b.numpy()[0]
    dof_a = position_solver.mf_dof_a.numpy()[0]
    dof_b = position_solver.mf_dof_b.numpy()[0]
    world_dofs = position_solver.world_dof_indices.numpy()[0]
    end_gaps = []
    for row in contact_rows:
        jv = 0.0
        if dof_a[row] >= 0:
            indices = world_dofs[dof_a[row] : dof_a[row] + 6]
            jv += float(np.dot(J_a[row][indices >= 0], v_position[indices[indices >= 0]]))
        if dof_b[row] >= 0:
            indices = world_dofs[dof_b[row] : dof_b[row] + 6]
            jv += float(np.dot(J_b[row][indices >= 0], v_position[indices[indices >= 0]]))
        end_gaps.append(float(phi[row] + SIM_DT * jv))
    max_end_gap = max(end_gaps, default=0.0)
    return before, float(state_1.body_qd.numpy()[body][2]), n, position_impulse, max_phi, max_end_gap


def test_rhs_classifies_exact_end_gap_without_impulse(test, device):
    """Treat exactly and numerically reached surfaces as active without consulting impulse."""
    count = wp.array([1], dtype=wp.int32, device=device)
    dof_a = wp.array([[0]], dtype=wp.int32, device=device)
    dof_b = wp.array([[-1]], dtype=wp.int32, device=device)
    J_a = wp.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=wp.float32, device=device)
    J_b = wp.zeros((1, 1, 6), dtype=wp.float32, device=device)
    world_dofs = wp.array([[0, 1, 2, 3, 4, 5]], dtype=wp.int32, device=device)
    phi = wp.array([[1.0]], dtype=wp.float32, device=device)
    row_type = wp.array([[0]], dtype=wp.int32, device=device)
    target_velocity = wp.zeros((1, 1), dtype=wp.float32, device=device)
    row_restitution = wp.zeros((1, 1), dtype=wp.float32, device=device)
    position_velocity = wp.zeros((6,), dtype=wp.float32, device=device)
    rhs = wp.zeros((1, 1), dtype=wp.float32, device=device)

    def classify(normal_velocity, target=0.0, has_target=0):
        position_velocity.assign(np.array([normal_velocity, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        target_velocity.assign(np.array([[target]], dtype=np.float32))
        wp.launch(
            compute_mf_velocity_rhs,
            dim=1,
            inputs=[
                count,
                dof_a,
                dof_b,
                J_a,
                J_b,
                world_dofs,
                phi,
                row_type,
                target_velocity,
                row_restitution,
                has_target,
                0.5,
                position_velocity,
                position_velocity,
                0.5,
                1,
            ],
            outputs=[rhs],
            device=device,
        )
        return float(rhs.numpy()[0, 0])

    outside_velocity = np.float32(-1.999997)
    outside_end_gap = np.float32(1.0) + np.float32(0.5) * outside_velocity
    near_velocity = np.float32(-1.999999)
    near_end_gap = np.float32(1.0) + np.float32(0.5) * near_velocity

    test.assertGreater(float(outside_end_gap), 1.0e-6, "outside test case did not clear the activation slop")
    test.assertGreater(float(near_end_gap), 0.0, "near-boundary test case did not retain a positive residual")
    test.assertLessEqual(float(near_end_gap), 1.0e-6, "near-boundary test case exceeded the activation slop")

    test.assertEqual(classify(-1.0), 2.0, "a row ending 0.5 m away lost its speculative allowance")
    test.assertEqual(
        classify(float(outside_velocity)),
        2.0,
        "a row ending beyond the numerical boundary lost its speculative allowance",
    )
    test.assertEqual(
        classify(float(near_velocity)),
        0.0,
        "a binding row with a small positive residual retained its speculative allowance",
    )
    test.assertEqual(classify(-2.0), 0.0, "a row reaching the surface retained its speculative allowance")
    test.assertEqual(
        classify(-1.0, target=1.0, has_target=1),
        -1.0,
        "prescribed target velocity was omitted or used with the wrong sign",
    )


def test_velocity_iterations_do_not_halt_a_falling_body(test, device):
    """Land a falling body on the plane with the velocity pass enabled."""
    for iterations in (2, 8):
        h, v, n, _r = _drop(device, iterations)
        _assert_landed(test, h, v, n, f"pgs_velocity_iterations={iterations}")


def test_velocity_iterations_match_the_position_only_landing(test, device):
    """Land at the same height with and without velocity iterations."""
    baseline, base_v, base_n, _r0 = _drop(device, 0)
    with_velocity, vel_v, vel_n, _r4 = _drop(device, 4)
    _assert_landed(test, baseline, base_v, base_n, "velocity_iterations=0")
    _assert_landed(test, with_velocity, vel_v, vel_n, "velocity_iterations=4")
    test.assertAlmostEqual(
        float(with_velocity[-1]),
        float(baseline[-1]),
        delta=0.002,
        msg=(
            f"resting height moved from {float(baseline[-1]):.4f} m to "
            f"{float(with_velocity[-1]):.4f} m when velocity iterations were enabled"
        ),
    )


def test_every_contact_response_route_lands(test, device):
    """Land correctly on the free-body and the dense contact rows.

    Asserts the exact ``contact_path`` each scene took, so the suite cannot claim
    coverage of a route the scene never reaches.
    """
    for scene in ("free", "articulated"):
        h, v, n, routes = _drop(device, 4, scene=scene)
        _assert_landed(test, h, v, n, scene)
        test.assertEqual(
            routes,
            EXPECTED_PATHS[scene],
            f"{scene}: exercised contact_path {sorted(routes)}, expected {sorted(EXPECTED_PATHS[scene])}",
        )


def test_non_crossing_row_keeps_its_speculative_allowance(test, device):
    """Leave the velocity untouched when the body cannot reach the surface.

    With ``phi + h*u > 0`` the position trajectory remains separated. If
    the velocity pass dropped its ``phi/h`` allowance it would forbid any
    approach and brake a body still far from contact.
    """
    gap = 0.02
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 0.5 * gap / SIM_DT)
    test.assertLess(before, -1.0, "setup failed to give the body an approach velocity")
    test.assertGreater(n, 0, "no speculative contact was generated, so the branch was never reached")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    test.assertEqual(impulse, 0.0, f"a non-crossing row took a position impulse of {impulse:.3e}")
    test.assertGreater(end_gap, 0.0, f"non-crossing row ended at gap {end_gap:+.3e} m")
    test.assertAlmostEqual(
        after, before, delta=0.02 * abs(before), msg=f"free approach was braked: {before:.3f} -> {after:.3f} m/s"
    )


def test_crossing_row_loses_its_speculative_allowance(test, device):
    """Arrest a body that would cross the surface within the step.

    A blanket "always retain phi/h" implementation leaves it closing at the gap
    rate instead, which this bound catches.
    """
    gap = 0.002
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 5.0 * gap / SIM_DT)
    test.assertLess(before, -1.0, "setup failed to give the body an approach velocity")
    test.assertGreater(n, 0, "no contact was generated")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    test.assertGreater(impulse, 0.0, "a crossing row took no position impulse")
    test.assertLessEqual(end_gap, 1.0e-6, f"crossing row still ended at gap {end_gap:+.3e} m")
    # Magnitude, not a signed bound: "after <= 0.05*|before|" is satisfied by any
    # negative value, so a row that merely slowed from -20 to -4 m/s would pass.
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"crossing contact kept approaching: {before:.3f} -> {after:.3f} m/s",
    )


def test_light_body_crossing_uses_end_gap_not_impulse_scale(test, device):
    """Arrest a very light crossing body independently of impulse scale.

    The impulse is effective mass times the velocity change, so it can be made
    arbitrarily small without changing the physics of the impact. At this mass it
    is ~2e-11, below the 1e-9 absolute threshold an impulse classifier once
    used. End-gap classification is invariant to that mass scaling. The mass is
    deliberately unphysical: the point is the classifier, not the scenario.
    """
    gap = 0.002
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 5.0 * gap / SIM_DT, mass=1.0e-12)
    test.assertGreater(n, 0, "no contact was generated")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    # Straddles the 1e-9 cutoff this implementation once used: positive, so the
    # row reaches the surface, yet small enough that an absolute threshold misses it.
    test.assertGreater(impulse, 0.0, "light crossing row took no position impulse")
    test.assertLess(impulse, 1.0e-9, f"position impulse {impulse:.3e} does not straddle the old 1e-9 cutoff")
    test.assertLessEqual(end_gap, 1.0e-6, f"light crossing row still ended at gap {end_gap:+.3e} m")
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"light crossing body kept approaching: {before:.3f} -> {after:.3f} m/s (position impulse {impulse:.3e})",
    )


def test_multiworld_end_gap_uses_each_world_position_velocity(test, device):
    """Classify speculative rows from the corresponding world's velocity."""
    gap = 0.002
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0)
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + gap), wp.quat_identity()))
    template.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    template.add_ground_plane(cfg=cfg)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    model = builder.finalize(device=device)
    solver = _make_solver(model, pgs_velocity_iterations=4)
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy().reshape(2, 6)
    joint_qd[0, 2] = -0.5 * gap / SIM_DT
    joint_qd[1, 2] = -5.0 * gap / SIM_DT
    state_0.joint_qd.assign(joint_qd.reshape(-1))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    state_0.clear_forces()
    solver.step(state_0, state_1, model.control(), contacts, SIM_DT)
    after = state_1.body_qd.numpy()[:, 2]
    test.assertAlmostEqual(
        float(after[0]),
        float(joint_qd[0, 2]),
        delta=0.02 * abs(float(joint_qd[0, 2])),
        msg=f"non-crossing world was braked to {after[0]:.3f} m/s",
    )
    test.assertLess(
        abs(float(after[1])),
        0.05 * abs(float(joint_qd[1, 2])),
        f"crossing world kept approaching at {after[1]:.3f} m/s",
    )


def test_warm_start_with_velocity_iterations_is_supported(test, device):
    """Combine the contact warm start with end-gap classification on both row families."""
    # Exercise the dense and free-body identity warm starts through the velocity
    # iterations, not just their construction.
    h, v, n, routes = _drop(device, 4, scene="articulated", pgs_warmstart=True)
    _assert_landed(test, h, v, n, "dense warm start")
    test.assertEqual(routes, {PATH_DENSE})

    h, v, n, routes = _drop(device, 4, pgs_warmstart=True)
    _assert_landed(test, h, v, n, "free-body warm start")
    test.assertEqual(routes, {PATH_MATRIX_FREE})


devices = get_selected_cuda_test_devices()


class TestFeatherPGSVelocityPassSpeculative(unittest.TestCase):
    pass


for _fn in (
    test_rhs_classifies_exact_end_gap_without_impulse,
    test_velocity_iterations_do_not_halt_a_falling_body,
    test_velocity_iterations_match_the_position_only_landing,
    test_every_contact_response_route_lands,
    test_non_crossing_row_keeps_its_speculative_allowance,
    test_crossing_row_loses_its_speculative_allowance,
    test_light_body_crossing_uses_end_gap_not_impulse_scale,
    test_multiworld_end_gap_uses_each_world_position_velocity,
    test_warm_start_with_velocity_iterations_is_supported,
):
    add_function_test(TestFeatherPGSVelocityPassSpeculative, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
