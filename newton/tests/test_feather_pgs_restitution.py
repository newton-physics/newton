# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical CUDA tests for FeatherPGS rigid-contact restitution.

The ordinary normal contact row freezes the unconstrained incident velocity
before solving and enforces Newton's impact law

    u_n^+ = -e u_n^-

for a contact predicted to reach the surface during the step. These tests use
one-step, frictionless scenes so the expected velocity and energy are
closed-form rather than inferred from a later bounce height.
"""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

RADIUS = 0.05
CONTACT_GAP = 0.1
DEFAULT_DT = 1.0e-3
IMPACT_FRACTION = 0.4

# Values written into ``SolverFeatherPGS.contact_path``.
PATH_DENSE = 0
PATH_MATRIX_FREE = 1

# The route each scene takes: a lone free body uses the free-body rows, and a
# contact on an articulated root uses the dense rows.
EXPECTED_PATHS = {"free": {PATH_MATRIX_FREE}, "articulated": {PATH_DENSE}}


def _diagonal_inertia(value):
    """Return an isotropic inertia tensor."""
    return wp.mat33(value, 0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0, value)


def _sphere_inertia(mass):
    """Return the inertia of a solid sphere about its centre."""
    return _diagonal_inertia(0.4 * mass * RADIUS * RADIUS)


def _shape_cfg(restitution):
    """Return an inertialess, frictionless test material."""
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=0.0,
        restitution=restitution,
        margin=0.0,
        gap=CONTACT_GAP,
    )


def _build_plane_model(
    device,
    *,
    separation,
    restitution,
    plane_restitution=None,
    mass=1.0,
    gravity=(0.0, 0.0, 0.0),
    scene="free",
):
    """Build one sphere above a plane, optionally as a two-link articulation."""
    builder = newton.ModelBuilder(gravity=gravity)
    xform = wp.transform(wp.vec3(0.0, 0.0, RADIUS + separation), wp.quat_identity())
    if scene == "free":
        body = builder.add_body(
            xform=xform,
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
    elif scene == "articulated":
        # A concentric, shape-free fixed child makes this a real floating
        # articulation without adding a second contact or a changing contact
        # point velocity to the analytical one-row problem.
        body = builder.add_link(
            xform=xform,
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
        child_mass = 0.25 * mass
        child = builder.add_link(
            xform=xform,
            mass=child_mass,
            inertia=_sphere_inertia(child_mass),
            lock_inertia=True,
        )
        root_joint = builder.add_joint_free(child=body)
        fixed_joint = builder.add_joint_fixed(parent=body, child=child)
        builder.add_articulation([root_joint, fixed_joint])
    else:
        raise ValueError(f"unknown scene {scene!r}")

    builder.add_shape_sphere(body, radius=RADIUS, cfg=_shape_cfg(restitution))
    builder.add_ground_plane(cfg=_shape_cfg(restitution if plane_restitution is None else plane_restitution))
    return builder.finalize(device=device), body


def _make_solver(
    model,
    *,
    velocity_iterations=0,
    restitution_velocity_threshold=0.0,
    pgs_iterations=16,
    pgs_warmstart=False,
):
    """Construct a deterministic frictionless FeatherPGS test solver without angular damping."""
    kwargs = {
        "pgs_iterations": pgs_iterations,
        "pgs_velocity_iterations": velocity_iterations,
        "pgs_cfm": 0.0,
        "pgs_omega": 1.0,
        "pgs_warmstart": pgs_warmstart,
    }
    if restitution_velocity_threshold is not None:
        kwargs["restitution_velocity_threshold"] = restitution_velocity_threshold
    solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs)
    solver.rigid_body_angular_damping.zero_()
    return solver


def _reset_state(model, state, vertical_velocity):
    """Restore the authored pose and set a free root's world-z velocity."""
    wp.copy(state.joint_q, model.joint_q)
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[2] = vertical_velocity
    state.joint_qd.assign(joint_qd)
    state.clear_forces()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _step_plane(
    device,
    *,
    separation,
    speed,
    restitution,
    dt=DEFAULT_DT,
    mass=1.0,
    plane_restitution=None,
    gravity=(0.0, 0.0, 0.0),
    initial_vertical_velocity=None,
    scene="free",
    velocity_iterations=0,
    restitution_velocity_threshold=0.0,
    pgs_iterations=16,
):
    """Run one sphere-plane step and return observable contact and state data."""
    model, body = _build_plane_model(
        device,
        separation=separation,
        restitution=restitution,
        plane_restitution=plane_restitution,
        mass=mass,
        gravity=gravity,
        scene=scene,
    )
    solver = _make_solver(
        model,
        velocity_iterations=velocity_iterations,
        restitution_velocity_threshold=restitution_velocity_threshold,
        pgs_iterations=pgs_iterations,
    )
    state_in, state_out = model.state(), model.state()
    vz = -speed if initial_vertical_velocity is None else initial_vertical_velocity
    _reset_state(model, state_in, vz)
    before = float(state_in.body_qd.numpy()[body][2])

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, dt)

    count = int(contacts.rigid_contact_count.numpy()[0])
    active_paths = {int(v) for v in solver.contact_path.numpy()[:count] if v >= 0}
    qd = state_out.body_qd.numpy()[body].astype(np.float64)
    q = state_out.body_q.numpy()[body].astype(np.float64)
    return {
        "before": before,
        "after": float(qd[2]),
        "body_qd": qd,
        "end_gap": float(q[2] - RADIUS),
        "contact_count": count,
        "paths": active_paths,
    }


def _assert_velocity(test, actual, expected, scale, label, *, rtol=2.0e-3, atol=2.0e-4):
    """Assert a scalar velocity with a scale-aware float32 tolerance."""
    test.assertAlmostEqual(
        actual,
        expected,
        delta=atol + rtol * max(abs(scale), abs(expected)),
        msg=f"{label}: got {actual:+.7f} m/s, expected {expected:+.7f} m/s",
    )


def _assert_crossing_setup(test, result, separation, speed, dt, label):
    """Prove that a test generated a positive-gap contact crossing this step."""
    test.assertGreater(separation, 0.0, f"{label}: contact did not start separated")
    test.assertLess(separation, speed * dt, f"{label}: contact could not reach the surface this step")
    test.assertEqual(result["contact_count"], 1, f"{label}: expected exactly one contact")
    _assert_velocity(test, result["before"], -speed, speed, f"{label}, incident state", rtol=1.0e-6, atol=1.0e-6)


def _assert_restitution_row_position(test, result, separation, speed, restitution, dt, label):
    """Assert the single-state PGS position law and its known time-of-impact error."""
    expected_gap = separation + restitution * speed * dt
    test.assertAlmostEqual(
        result["end_gap"],
        expected_gap,
        delta=max(5.0e-6, 3.0e-3 * expected_gap),
        msg=f"{label}: ordinary restitution row used an unexpected position trajectory",
    )
    physical_gap = restitution * (speed * dt - separation)
    expected_error = (1.0 + restitution) * separation
    test.assertAlmostEqual(
        result["end_gap"] - physical_gap,
        expected_error,
        delta=max(8.0e-6, 4.0e-3 * expected_error),
        msg=f"{label}: phase-dependent position error changed unexpectedly",
    )


def test_one_step_restitution_law_over_coefficient_speed_and_mass(test, device):
    """Enforce Newton's impact law across coefficient, speed, and mass scales."""
    cases = []
    cases.extend((f"e={e}", e, 4.0, 1.0) for e in (0.25, 0.6, 1.0))
    cases.extend((f"speed={speed}", 0.6, speed, 1.0) for speed in (0.05, 2.0, 20.0))
    cases.extend((f"mass={mass}", 0.6, 4.0, mass) for mass in (1.0e-6, 1.0, 1.0e6))

    for label, restitution, speed, mass in cases:
        with test.subTest(case=label):
            separation = IMPACT_FRACTION * speed * DEFAULT_DT
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                mass=mass,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            expected = restitution * speed
            _assert_velocity(test, result["after"], expected, speed, label)
            _assert_restitution_row_position(test, result, separation, speed, restitution, DEFAULT_DT, label)
            energy_ratio = (result["after"] / speed) ** 2
            test.assertAlmostEqual(
                energy_ratio,
                restitution * restitution,
                delta=5.0e-3,
                msg=f"{label}: kinetic-energy ratio was {energy_ratio:.6f}",
            )


def test_crossing_result_is_invariant_to_dt_and_time_of_impact(test, device):
    """Use incident speed rather than gap-clipped velocity at every impact time."""
    restitution = 0.65
    speed = 3.0
    cases = (
        # At impact_fraction -> 0 the single-state position law coincides with
        # the analytical rebound (both e*speed*dt), so the first case anchors
        # the position assertions to exact ground truth; the rest pin the law
        # and its known (1+e)*separation deviation across the step phase.
        (1.0e-3, 0.01),
        (2.5e-4, 0.1),
        (2.5e-4, 0.9),
        (1.0e-3, 0.5),
        (1.0e-2, 0.1),
        (1.0e-2, 0.9),
    )
    for pgs_iterations in (1, 2, 8):
        for dt, impact_fraction in cases:
            label = f"iters={pgs_iterations}, dt={dt:g}, impact/dt={impact_fraction:g}"
            with test.subTest(pgs_iterations=pgs_iterations, dt=dt, impact_fraction=impact_fraction):
                separation = speed * dt * impact_fraction
                result = _step_plane(
                    device,
                    separation=separation,
                    speed=speed,
                    restitution=restitution,
                    dt=dt,
                    pgs_iterations=pgs_iterations,
                )
                _assert_crossing_setup(test, result, separation, speed, dt, label)
                _assert_velocity(test, result["after"], restitution * speed, speed, label)
                _assert_restitution_row_position(test, result, separation, speed, restitution, dt, label)


def test_penetrating_impact_replaces_baumgarte_with_restitution(test, device):
    """Keep penetration correction out of the qualifying restitution target."""
    restitution = 0.5
    speed = 2.0
    for penetration in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
        label = f"penetration={penetration:g}"
        with test.subTest(penetration=penetration):
            result = _step_plane(
                device,
                separation=-penetration,
                speed=speed,
                restitution=restitution,
                velocity_iterations=0,
                pgs_iterations=8,
            )
            test.assertEqual(result["contact_count"], 1, f"{label}: expected exactly one contact")
            test.assertEqual(result["paths"], {PATH_MATRIX_FREE}, f"{label}: wrong contact route")
            _assert_velocity(test, result["before"], -speed, speed, f"{label}, incident state")
            _assert_velocity(test, result["after"], restitution * speed, speed, label)


def test_restitution_uses_the_post_force_incident_predictor(test, device):
    """Include this step's known gravity impulse in the frozen incident speed."""
    restitution = 0.6
    initial_speed = 2.0
    gravity = 9.81
    dt = 1.0e-2
    incident_speed = initial_speed + gravity * dt
    separation = IMPACT_FRACTION * incident_speed * dt
    result = _step_plane(
        device,
        separation=separation,
        speed=initial_speed,
        restitution=restitution,
        dt=dt,
        gravity=(0.0, 0.0, -gravity),
    )

    test.assertEqual(result["contact_count"], 1, "gravity case generated no unique contact")
    _assert_velocity(test, result["before"], -initial_speed, initial_speed, "gravity, authored velocity")
    _assert_restitution_row_position(
        test, result, separation, incident_speed, restitution, dt, "gravity, post-force incident predictor"
    )
    _assert_velocity(
        test,
        result["after"],
        restitution * incident_speed,
        incident_speed,
        "gravity, post-force incident predictor",
    )


def test_non_crossing_speculative_contact_does_not_bounce(test, device):
    """Preserve free approach when a speculative contact cannot impact this step."""
    restitution = 1.0
    speed = 3.0
    separation = 1.25 * speed * DEFAULT_DT
    result = _step_plane(
        device,
        separation=separation,
        speed=speed,
        restitution=restitution,
    )
    test.assertGreater(result["contact_count"], 0, "the speculative contact was never generated")
    test.assertGreater(separation - speed * DEFAULT_DT, 0.0, "the setup unexpectedly crosses the surface")
    _assert_velocity(test, result["after"], -speed, speed, "non-crossing contact")
    test.assertGreater(result["end_gap"], 0.0, "the non-crossing body reached or crossed the plane")


def test_shape_restitution_uses_symmetric_arithmetic_average(test, device):
    """Average sanitized shape coefficients symmetrically, including authored zero."""
    speed = 4.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    cases = (
        (0.0, 0.8, 0.4),
        (0.2, 0.8, 0.5),
        (0.8, 0.2, 0.5),
        (2.0, 0.8, 0.9),
        (-0.2, 0.8, 0.4),
        (float("nan"), 0.8, 0.4),
    )
    for sphere_e, plane_e, mixed in cases:
        label = f"sphere={sphere_e}, plane={plane_e}"
        with test.subTest(sphere=sphere_e, plane=plane_e):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=sphere_e,
                plane_restitution=plane_e,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            _assert_velocity(test, result["after"], mixed * speed, speed, label)


def test_restitution_velocity_threshold_uses_incident_relative_speed(test, device):
    """Suppress low-speed bounce and activate immediately above the threshold."""
    threshold = 0.5
    restitution = 0.8
    for speed in (0.25, 0.49, 0.51, 2.0):
        label = f"speed={speed}, threshold={threshold}"
        with test.subTest(speed=speed):
            separation = IMPACT_FRACTION * speed * DEFAULT_DT
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                restitution_velocity_threshold=threshold,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            expected = restitution * speed if speed > threshold else -separation / DEFAULT_DT
            _assert_velocity(test, result["after"], expected, speed, label)
            if speed > threshold:
                _assert_restitution_row_position(test, result, separation, speed, restitution, DEFAULT_DT, label)
            else:
                test.assertLess(abs(result["end_gap"]), 5.0e-6, f"{label}: speculative row missed the surface")

    # Both bodies move quickly in the world frame, but their closing speed is
    # below the threshold. This catches a gate applied to either body's
    # absolute speed instead of the contact's relative normal speed.
    out_a, out_b, _distance, count, _paths = _step_two_spheres(
        device,
        mass_a=1.0,
        mass_b=1.0,
        velocity_a=10.0,
        velocity_b=9.6,
        restitution=restitution,
        restitution_velocity_threshold=threshold,
    )
    test.assertEqual(count, 1, "relative-threshold test generated no unique contact")
    # Restitution is suppressed, but the pair still closes the remaining gap
    # under the speculative allowance and arrives touching with relative speed
    # separation/dt, the impulse shared equally between the equal masses -- the
    # same rule the single-body sub-cases assert above. Expecting a common 9.8
    # would freeze the bodies mid-air while still separated.
    residual_closing = IMPACT_FRACTION * (10.0 - 9.6)
    _assert_velocity(test, out_a, 9.8 + 0.5 * residual_closing, 0.4, "relative threshold, body A")
    _assert_velocity(test, out_b, 9.8 - 0.5 * residual_closing, 0.4, "relative threshold, body B")
    test.assertAlmostEqual(out_a + out_b, 19.6, delta=1.0e-4, msg="relative threshold: momentum drifted")


def test_largest_finite_threshold_disables_restitution_without_material_edits(test, device):
    """Give the zero-restitution result with a float32-maximum threshold, leaving materials unchanged.

    This is the solver-wide way to turn restitution off; an infinite threshold is
    rejected, and zeroing one shape's coefficient still leaves the average.
    """
    speed = 2.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    largest = float(np.finfo(np.float32).max)
    for scene in ("free", "articulated"):
        with test.subTest(scene=scene):
            common = {"separation": separation, "speed": speed, "scene": scene}
            bouncing = _step_plane(device, restitution=0.8, plane_restitution=0.8, **common)
            one_zero = _step_plane(device, restitution=0.0, plane_restitution=0.8, **common)
            inelastic = _step_plane(device, restitution=0.0, plane_restitution=0.0, **common)
            disabled = _step_plane(
                device, restitution=0.8, plane_restitution=0.8, restitution_velocity_threshold=largest, **common
            )
            _assert_velocity(test, bouncing["after"], 0.8 * speed, speed, f"{scene}: authored rebound")
            _assert_velocity(test, one_zero["after"], 0.4 * speed, speed, f"{scene}: averaged rebound")
            test.assertAlmostEqual(disabled["after"], inelastic["after"], delta=1.0e-6 * speed)
            test.assertAlmostEqual(disabled["end_gap"], inelastic["end_gap"], delta=1.0e-7)


def test_bouncing_ball_settles_to_rest_under_gravity(test, device):
    """Settle a bouncing ball on the surface without perpetual micro-bounce or sinking.

    Restitution shares the contact row with the speculative/Baumgarte law, so
    the long-run interaction with resting contact is a distinct failure mode
    from any single-impact assertion: a bounce target that keeps firing at
    resting speeds produces a Zeno micro-bounce, while an over-eager gate
    lets the row sink. Impacts decay 2.0 -> 1.0 -> 0.5 m/s, the third landing
    falls at the 0.5 m/s threshold and is suppressed, and the tail of the
    rollout must be quiescent on the surface.
    """
    restitution = 0.5
    speed = 2.0
    gravity = 9.81
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    model, body = _build_plane_model(
        device,
        separation=separation,
        restitution=restitution,
        gravity=(0.0, 0.0, -gravity),
    )
    solver = _make_solver(model, restitution_velocity_threshold=0.5)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -speed)

    heights, speeds = [], []
    for _ in range(800):
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
        state_in, state_out = state_out, state_in
        heights.append(float(state_in.body_q.numpy()[body][2]))
        speeds.append(float(state_in.body_qd.numpy()[body][2]))
    heights = np.array(heights)
    speeds = np.array(speeds)

    # Non-vacuity: the first impact must actually rebound at ~e*speed.
    test.assertGreater(float(speeds[:100].max()), 0.9 * restitution * speed, "the ball never bounced")
    test.assertLess(float(heights.min()), RADIUS + 1.0e-3, "the ball never reached the surface")
    test.assertGreater(float(heights.min()), RADIUS - 2.0e-3, "the ball sank through the plane")
    # Quiescence: by 0.6 s every impact is below the threshold, so the tail
    # must rest on the surface instead of bouncing or creeping.
    tail_heights = heights[-200:]
    tail_speeds = speeds[-200:]
    test.assertLess(float(np.abs(tail_speeds).max()), 0.1, "resting contact kept bouncing")
    test.assertLess(float(tail_heights.max()), RADIUS + 1.5e-3, "resting contact hovered above the surface")
    test.assertGreater(float(tail_heights.min()), RADIUS - 1.5e-3, "resting contact sank into the plane")


def _build_two_sphere_model(device, *, separation, restitution, mass_a, mass_b):
    """Build two free spheres closing along their line of centres."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = _shape_cfg(restitution)
    body_a = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=mass_a,
        inertia=_sphere_inertia(mass_a),
        lock_inertia=True,
    )
    body_b = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * RADIUS + separation), wp.quat_identity()),
        mass=mass_b,
        inertia=_sphere_inertia(mass_b),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body_a, radius=RADIUS, cfg=cfg)
    builder.add_shape_sphere(body_b, radius=RADIUS, cfg=cfg)
    return builder.finalize(device=device), body_a, body_b


def _step_two_spheres(
    device,
    *,
    mass_a,
    mass_b,
    velocity_a,
    velocity_b,
    restitution,
    restitution_velocity_threshold=0.0,
):
    """Run one two-body impact and return the final normal velocities."""
    closing_speed = velocity_a - velocity_b
    separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
    model, body_a, body_b = _build_two_sphere_model(
        device,
        separation=separation,
        restitution=restitution,
        mass_a=mass_a,
        mass_b=mass_b,
    )
    solver = _make_solver(model, restitution_velocity_threshold=restitution_velocity_threshold)
    state_in, state_out = model.state(), model.state()
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[2] = velocity_a
    joint_qd[8] = velocity_b
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
    qd = state_out.body_qd.numpy()
    q = state_out.body_q.numpy()
    count = int(contacts.rigid_contact_count.numpy()[0])
    paths = {int(v) for v in solver.contact_path.numpy()[:count] if v >= 0}
    return float(qd[body_a][2]), float(qd[body_b][2]), float(q[body_b][2] - q[body_a][2]), count, paths


def test_two_body_impact_conserves_momentum_and_reverses_relative_speed(test, device):
    """Match the exact two-body impact solution over mass ratio and frame shift."""
    restitution = 0.6
    for mass_a, mass_b, frame_speed in ((1.0, 1.0, 0.0), (1.0, 4.0, 0.0), (4.0, 1.0, 0.0), (1.0, 4.0, 3.0)):
        velocity_a = 2.0 + frame_speed
        velocity_b = -1.0 + frame_speed
        label = f"m=({mass_a},{mass_b}), frame={frame_speed}"
        with test.subTest(mass_a=mass_a, mass_b=mass_b, frame_speed=frame_speed):
            out_a, out_b, distance, count, paths = _step_two_spheres(
                device,
                mass_a=mass_a,
                mass_b=mass_b,
                velocity_a=velocity_a,
                velocity_b=velocity_b,
                restitution=restitution,
            )
            total_mass = mass_a + mass_b
            expected_a = (
                (mass_a - restitution * mass_b) * velocity_a + (1.0 + restitution) * mass_b * velocity_b
            ) / total_mass
            expected_b = (
                (1.0 + restitution) * mass_a * velocity_a + (mass_b - restitution * mass_a) * velocity_b
            ) / total_mass
            test.assertEqual(count, 1, f"{label}: expected exactly one sphere-sphere contact")
            test.assertEqual(paths, {PATH_MATRIX_FREE}, f"{label}: unexpected contact route {paths}")
            closing_speed = velocity_a - velocity_b
            separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
            expected_distance = 2.0 * RADIUS + separation + restitution * closing_speed * DEFAULT_DT
            test.assertAlmostEqual(distance, expected_distance, delta=2.0e-4, msg=f"{label}: wrong end separation")
            _assert_velocity(test, out_a, expected_a, velocity_a - velocity_b, f"{label}, body A")
            _assert_velocity(test, out_b, expected_b, velocity_a - velocity_b, f"{label}, body B")

            momentum_before = mass_a * velocity_a + mass_b * velocity_b
            momentum_after = mass_a * out_a + mass_b * out_b
            test.assertAlmostEqual(
                momentum_after,
                momentum_before,
                delta=2.0e-3 * max(1.0, abs(momentum_before)),
                msg=f"{label}: normal momentum changed",
            )
            test.assertAlmostEqual(
                out_b - out_a,
                -restitution * (velocity_b - velocity_a),
                delta=5.0e-3,
                msg=f"{label}: relative impact law failed",
            )
            reduced_mass = mass_a * mass_b / total_mass
            expected_energy_loss = 0.5 * reduced_mass * (1.0 - restitution**2) * (velocity_a - velocity_b) ** 2
            energy_before = 0.5 * mass_a * velocity_a**2 + 0.5 * mass_b * velocity_b**2
            energy_after = 0.5 * mass_a * out_a**2 + 0.5 * mass_b * out_b**2
            test.assertAlmostEqual(
                energy_before - energy_after,
                expected_energy_loss,
                delta=5.0e-3 * max(1.0, energy_before),
                msg=f"{label}: collision energy loss was inconsistent with e",
            )


def test_restitution_is_relative_to_a_moving_kinematic_surface(test, device):
    """Measure rebound relative to prescribed motion on the free-body rows."""
    restitution = 0.6
    closing_speed = 3.0
    separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
    for frame_speed in (0.0, 7.0):
        label = f"frame={frame_speed}"
        with test.subTest(frame_speed=frame_speed):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            cfg = _shape_cfg(restitution)
            dynamic_body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=_sphere_inertia(1.0),
                lock_inertia=True,
            )
            kinematic_body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * RADIUS + separation), wp.quat_identity()),
                mass=1.0,
                inertia=_sphere_inertia(1.0),
                lock_inertia=True,
                is_kinematic=True,
            )
            builder.add_shape_sphere(dynamic_body, radius=RADIUS, cfg=cfg)
            builder.add_shape_sphere(kinematic_body, radius=RADIUS, cfg=cfg)
            model = builder.finalize(device=device)
            solver = _make_solver(model)
            state_in, state_out = model.state(), model.state()
            velocity_dynamic = frame_speed + 2.0
            velocity_surface = frame_speed - 1.0
            joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
            joint_qd[2] = velocity_dynamic
            joint_qd[8] = velocity_surface
            state_in.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

            count = int(contacts.rigid_contact_count.numpy()[0])
            paths = {int(v) for v in solver.contact_path.numpy()[:count] if v >= 0}
            velocity_after = float(state_out.body_qd.numpy()[dynamic_body][2])
            expected_after = velocity_surface - restitution * closing_speed
            test.assertEqual(count, 1, f"{label}: expected one sphere contact")
            test.assertEqual(paths, {PATH_MATRIX_FREE}, f"{label}: wrong contact route")
            _assert_velocity(test, velocity_after, expected_after, closing_speed, label, rtol=5.0e-3)
            test.assertAlmostEqual(
                velocity_surface - velocity_after,
                restitution * closing_speed,
                delta=1.5e-2,
                msg=f"{label}: restitution was not measured relative to the moving surface",
            )


def test_every_contact_response_route_enforces_restitution(test, device):
    """Enforce the same impact law on the free-body and the dense contact rows."""
    restitution = 0.65
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for scene in ("free", "articulated"):
        with test.subTest(scene=scene):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                scene=scene,
                velocity_iterations=0,
                pgs_iterations=16,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, scene)
            test.assertEqual(result["paths"], EXPECTED_PATHS[scene], f"{scene}: wrong route")
            _assert_velocity(test, result["after"], restitution * speed, speed, scene, rtol=5.0e-3)


def test_multiworld_restitution_keeps_case_data_isolated(test, device):
    """Apply each world's coefficient and crossing state without cross-world leakage."""
    # (e, speed, mass, toi/dt); the third case is deliberately non-crossing.
    cases = ((0.25, 2.0, 1.0, 0.4), (0.75, 4.0, 10.0, 0.7), (1.0, 3.0, 0.5, 1.25), (0.0, 5.0, 2.0, 0.2))
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for restitution, speed, mass, impact_fraction in cases:
        world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        separation = speed * DEFAULT_DT * impact_fraction
        body = world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + separation), wp.quat_identity()),
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
        cfg = _shape_cfg(restitution)
        world.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        world.add_ground_plane(cfg=cfg)
        scene.add_world(world)
    model = scene.finalize(device=device)
    solver = _make_solver(model)
    state_in, state_out = model.state(), model.state()
    joint_qd = np.zeros((len(cases), 6), dtype=np.float32)
    joint_qd[:, 2] = [-case[1] for case in cases]
    state_in.joint_qd.assign(joint_qd.reshape(-1))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

    count = int(contacts.rigid_contact_count.numpy()[0])
    after = state_out.body_qd.numpy()[:, 2]
    end_gap = state_out.body_q.numpy()[:, 2] - RADIUS
    paths = solver.contact_path.numpy()[:count]
    test.assertEqual(count, len(cases), "expected exactly one contact in every world")
    test.assertTrue(np.all(paths == PATH_MATRIX_FREE), f"multiworld contacts used paths {paths.tolist()}")
    for world, (restitution, speed, _mass, impact_fraction) in enumerate(cases):
        with test.subTest(world=world):
            crossing = impact_fraction < 1.0
            if crossing and restitution > 0.0:
                expected = restitution * speed
            elif crossing:
                expected = -impact_fraction * speed
            else:
                expected = -speed
            _assert_velocity(test, float(after[world]), expected, speed, f"world {world}")
            if crossing and restitution > 0.0:
                expected_gap = impact_fraction * speed * DEFAULT_DT + restitution * speed * DEFAULT_DT
                test.assertAlmostEqual(
                    float(end_gap[world]), expected_gap, delta=2.0e-4, msg=f"world {world} wrong row trajectory"
                )
            elif crossing:
                test.assertLess(abs(float(end_gap[world])), 2.0e-4, f"world {world} missed the surface")
            else:
                test.assertGreater(float(end_gap[world]), 0.0, f"world {world} unexpectedly reached the surface")


def test_restitution_uses_ordinary_rows_and_validates_configuration(test, device):
    """Keep velocity iterations explicit while provisioning live restitution rows."""
    positive_model, _ = _build_plane_model(
        device,
        separation=1.0e-3,
        restitution=0.6,
    )
    solver = _make_solver(positive_model, velocity_iterations=0)
    test.assertEqual(solver.pgs_velocity_iterations, 0)
    test.assertFalse(hasattr(solver, "_velocity_post_iterations"), "zero iterations were silently overridden")
    test.assertEqual(solver.row_restitution.shape, (solver.world_count, solver.dense_max_constraints))
    test.assertEqual(solver.mf_row_restitution.shape, (solver.world_count, solver.mf_max_constraints))
    for invalid_threshold in (-1.0, float("nan"), float("inf")):
        with test.subTest(invalid_threshold=invalid_threshold):
            with test.assertRaisesRegex(ValueError, "restitution_velocity_threshold"):
                _make_solver(positive_model, restitution_velocity_threshold=invalid_threshold)

    zero_model, _ = _build_plane_model(
        device,
        separation=1.0e-3,
        restitution=0.0,
    )
    zero_solver = _make_solver(zero_model, velocity_iterations=0)
    test.assertEqual(zero_solver.pgs_velocity_iterations, 0)

    zero_model.shape_material_restitution.fill_(0.6)
    zero_solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    default_solver = _make_solver(positive_model, velocity_iterations=0, restitution_velocity_threshold=None)
    test.assertAlmostEqual(default_solver.restitution_velocity_threshold, 0.5)


def test_velocity_iteration_counts_reach_the_same_single_row_solution(test, device):
    """Keep restitution correct while optional velocity iterations refine the row."""
    restitution = 0.7
    speed = 2.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for iterations in (0, 1, 4, 12):
        with test.subTest(velocity_iterations=iterations):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                velocity_iterations=iterations,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, f"iterations={iterations}")
            _assert_velocity(test, result["after"], restitution * speed, speed, f"iterations={iterations}")


def test_warm_start_history_does_not_change_or_repeat_restitution(test, device):
    """Use the current incident velocity once under the contact warm start of both row families."""
    restitution = 0.6
    # Both incident speeds used below must cross from this same authored pose.
    separation = IMPACT_FRACTION * 2.0 * DEFAULT_DT
    cases = (
        ("free-body rows", "free", PATH_MATRIX_FREE),
        ("dense rows", "articulated", PATH_DENSE),
    )
    for label, scene, expected_path in cases:
        with test.subTest(warmstart=label):
            model, body = _build_plane_model(
                device,
                separation=separation,
                restitution=restitution,
                scene=scene,
            )
            solver = _make_solver(
                model,
                velocity_iterations=0,
                pgs_warmstart=True,
            )
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
            contacts = pipeline.contacts()
            state_in, state_out = model.state(), model.state()

            for speed in (2.0, 5.0):
                _reset_state(model, state_in, -speed)
                contacts.clear()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
                count = int(contacts.rigid_contact_count.numpy()[0])
                test.assertEqual(count, 1, f"{label}: reset impact generated {count} contacts")
                test.assertEqual(
                    {int(v) for v in solver.contact_path.numpy()[:count] if v >= 0},
                    {expected_path},
                    f"{label}: reset impact used the wrong route",
                )
                after = float(state_out.body_qd.numpy()[body][2])
                _assert_velocity(test, after, restitution * speed, speed, f"{label}, speed={speed}", rtol=5.0e-3)

            # The second impact ends on the plane and separates. A persistent
            # matched contact on the next step must not apply restitution a
            # second time from either warm-start impulse history or stale u^-.
            contacts.clear()
            pipeline.collide(state_out, contacts)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0, f"{label}: contact did not persist")
            solver.step(state_out, state_in, model.control(), contacts, DEFAULT_DT)
            separating = float(state_in.body_qd.numpy()[body][2])
            _assert_velocity(test, separating, restitution * 5.0, 5.0, f"{label}, separating step", rtol=1.0e-2)


def test_redundant_box_contacts_do_not_multiply_restitution_energy(test, device):
    """Bounce a flat box without row-count-dependent speed, torque, or energy gain."""
    hx, hy, hz = 0.1, 0.08, 0.05
    mass = 2.0
    restitution = 0.6
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    ix = mass * (hy * hy + hz * hz) / 3.0
    iy = mass * (hx * hx + hz * hz) / 3.0
    iz = mass * (hx * hx + hy * hy) / 3.0
    inertia = wp.mat33(ix, 0.0, 0.0, 0.0, iy, 0.0, 0.0, 0.0, iz)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, hz + separation), wp.quat_identity()),
        mass=mass,
        inertia=inertia,
        lock_inertia=True,
    )
    cfg = _shape_cfg(restitution)
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    solver = _make_solver(model, velocity_iterations=32, pgs_iterations=32)
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -speed)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

    count = int(contacts.rigid_contact_count.numpy()[0])
    qd = state_out.body_qd.numpy()[body].astype(np.float64)
    end_gap = float(state_out.body_q.numpy()[body][2] - hz)
    test.assertGreaterEqual(count, 4, f"box test generated only {count} contacts")
    _assert_velocity(test, float(qd[2]), restitution * speed, speed, "flat box", rtol=1.0e-2)
    test.assertLess(np.linalg.norm(qd[:2]), 1.0e-2, f"flat box gained tangential speed {qd[:2]}")
    test.assertLess(np.linalg.norm(qd[3:]), 2.0e-2, f"flat box gained angular speed {qd[3:]}")
    expected_gap = separation + restitution * speed * DEFAULT_DT
    test.assertAlmostEqual(end_gap, expected_gap, delta=3.0e-4, msg=f"flat box ended at gap {end_gap:+.3e} m")
    energy_before = 0.5 * mass * speed * speed
    energy_after = 0.5 * mass * float(np.dot(qd[:3], qd[:3])) + 0.5 * (
        ix * qd[3] ** 2 + iy * qd[4] ** 2 + iz * qd[5] ** 2
    )
    test.assertAlmostEqual(
        energy_after / energy_before,
        restitution * restitution,
        delta=2.0e-2,
        msg=f"flat-box energy ratio was {energy_after / energy_before:.6f}",
    )


def test_articulated_redundant_manifold_preserves_symmetric_rebound(test, device):
    """Bounce an articulated four-point foot without spurious spin or energy gain."""
    mass = 2.0
    child_mass = 0.25
    half = 0.08
    sphere_radius = 0.02
    foot_offset = 0.03
    restitution = 0.6
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    root_height = foot_offset + sphere_radius + separation
    inertia = _diagonal_inertia(0.02)

    for joint_kind in ("fixed", "revolute"):
        with test.subTest(joint=joint_kind):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            xform = wp.transform(wp.vec3(0.0, 0.0, root_height), wp.quat_identity())
            root = builder.add_link(xform=xform, mass=mass, inertia=inertia, lock_inertia=True)
            child = builder.add_link(
                xform=xform,
                mass=child_mass,
                inertia=_diagonal_inertia(0.002),
                lock_inertia=True,
            )
            root_joint = builder.add_joint_free(child=root)
            if joint_kind == "fixed":
                child_joint = builder.add_joint_fixed(parent=root, child=child)
            else:
                child_joint = builder.add_joint_revolute(
                    parent=root,
                    child=child,
                    axis=wp.vec3(0.0, 0.0, 1.0),
                )
            builder.add_articulation([root_joint, child_joint])
            cfg = _shape_cfg(restitution)
            for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)):
                builder.add_shape_sphere(
                    root,
                    xform=wp.transform(wp.vec3(sx * half, sy * half, -foot_offset), wp.quat_identity()),
                    radius=sphere_radius,
                    cfg=cfg,
                )
            builder.add_ground_plane(cfg=cfg)
            model = builder.finalize(device=device)
            solver = _make_solver(model, velocity_iterations=0, pgs_iterations=16)
            state_in, state_out = model.state(), model.state()
            _reset_state(model, state_in, -speed)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

            count = int(contacts.rigid_contact_count.numpy()[0])
            paths = {int(v) for v in solver.contact_path.numpy()[:count] if v >= 0}
            qd = state_out.body_qd.numpy()[root].astype(np.float64)
            expected_path = PATH_DENSE
            label = joint_kind
            test.assertGreaterEqual(count, 4, f"{label}: manifold had only {count} contacts")
            test.assertEqual(paths, {expected_path}, f"{label}: wrong contact route")
            _assert_velocity(test, float(qd[2]), restitution * speed, speed, label, rtol=7.5e-3)
            test.assertLess(np.linalg.norm(qd[3:]), 3.0e-2, f"{label}: symmetric foot spun at {qd[3:]}")


def test_cuda_graph_replay_reads_current_restitution_and_matches_eager(test, device):
    """Replay collide plus restitution with stable pointers and device-read material data."""
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT

    def make_fixture():
        model, body = _build_plane_model(
            device,
            separation=separation,
            restitution=0.0,
        )
        solver = _make_solver(model, velocity_iterations=0)
        state_in, state_out = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        control = model.control()

        def one_step():
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            # Keep addresses stable for graph replay.
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        return model, solver, body, state_in, contacts, one_step

    graph_model, graph_solver, graph_body, graph_state, graph_contacts, graph_step = make_fixture()
    eager_model, eager_solver, eager_body, eager_state, eager_contacts, eager_step = make_fixture()

    # Warm both fixtures through the same history, which also compiles the kernels.
    _reset_state(graph_model, graph_state, -speed)
    _reset_state(eager_model, eager_state, -speed)
    graph_step()
    eager_step()
    _reset_state(graph_model, graph_state, -speed)
    _reset_state(eager_model, eager_state, -speed)

    with wp.ScopedCapture(device) as capture:
        graph_step()

    # Change restitution from zero after capture. Row storage is independent
    # of construction-time values, and the coefficient must come from the
    # device array rather than a captured host scalar.
    for model, solver in ((graph_model, graph_solver), (eager_model, eager_solver)):
        model.shape_material_restitution.fill_(0.75)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    wp.capture_launch(capture.graph)
    eager_step()

    graph_velocity = float(graph_state.body_qd.numpy()[graph_body][2])
    eager_velocity = float(eager_state.body_qd.numpy()[eager_body][2])
    graph_count = int(graph_contacts.rigid_contact_count.numpy()[0])
    eager_count = int(eager_contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(graph_count, 1, "captured collision generated no unique contact")
    test.assertEqual(eager_count, 1, "eager collision generated no unique contact")
    _assert_velocity(test, graph_velocity, 0.75 * speed, speed, "captured step")
    test.assertAlmostEqual(graph_velocity, eager_velocity, delta=1.0e-6, msg="captured and eager steps diverged")


def test_notify_reads_replaced_and_in_place_restitution(test, device):
    """Match a fresh solver after a SHAPE_PROPERTIES notify, eagerly and in a captured graph."""
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for scene, old, replace, capture in itertools.product(
        ("free", "articulated"), (0.0, 1.0), (False, True), (False, True)
    ):
        new = 1.0 - old
        label = f"{scene} e={old}->{new} {'replaced' if replace else 'in place'} {'captured' if capture else 'eager'}"
        with test.subTest(label):
            model, body = _build_plane_model(device, separation=separation, restitution=old, scene=scene)
            solver = _make_solver(model)
            state_in, state_out = model.state(), model.state()
            _reset_state(model, state_in, -speed)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
            contacts = pipeline.contacts()
            control = model.control()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            if capture:
                with wp.ScopedCapture(device) as graph:
                    pipeline.collide(state_in, contacts)
                    solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            if replace:
                model.shape_material_restitution = wp.full(model.shape_count, new, dtype=float, device=device)
            else:
                model.shape_material_restitution.fill_(new)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, DEFAULT_DT)

            reference = model.state()
            _make_solver(model).step(state_in, reference, control, contacts, DEFAULT_DT)
            count = int(contacts.rigid_contact_count.numpy()[0])
            paths = {int(v) for v in solver.contact_path.numpy()[:count] if v >= 0}
            test.assertEqual(count, 1, f"{label}: expected one contact")
            test.assertEqual(paths, EXPECTED_PATHS[scene], f"{label}: wrong route")
            expected = float(reference.body_qd.numpy()[body][2])
            _assert_velocity(test, expected, new * speed if new else -separation / DEFAULT_DT, speed, label)
            test.assertAlmostEqual(float(state_out.body_qd.numpy()[body][2]), expected, delta=2.0e-5, msg=label)


devices = get_selected_cuda_test_devices()


class TestFeatherPGSRestitution(unittest.TestCase):
    pass


for _fn in (
    test_one_step_restitution_law_over_coefficient_speed_and_mass,
    test_crossing_result_is_invariant_to_dt_and_time_of_impact,
    test_penetrating_impact_replaces_baumgarte_with_restitution,
    test_restitution_uses_the_post_force_incident_predictor,
    test_non_crossing_speculative_contact_does_not_bounce,
    test_shape_restitution_uses_symmetric_arithmetic_average,
    test_restitution_velocity_threshold_uses_incident_relative_speed,
    test_largest_finite_threshold_disables_restitution_without_material_edits,
    test_bouncing_ball_settles_to_rest_under_gravity,
    test_two_body_impact_conserves_momentum_and_reverses_relative_speed,
    test_restitution_is_relative_to_a_moving_kinematic_surface,
    test_every_contact_response_route_enforces_restitution,
    test_multiworld_restitution_keeps_case_data_isolated,
    test_restitution_uses_ordinary_rows_and_validates_configuration,
    test_velocity_iteration_counts_reach_the_same_single_row_solution,
    test_warm_start_history_does_not_change_or_repeat_restitution,
    test_redundant_box_contacts_do_not_multiply_restitution_energy,
    test_articulated_redundant_manifold_preserves_symmetric_rebound,
    test_cuda_graph_replay_reads_current_restitution_and_matches_eager,
    test_notify_reads_replaced_and_in_place_restitution,
):
    add_function_test(TestFeatherPGSRestitution, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
