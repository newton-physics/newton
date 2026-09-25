# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for collision prediction horizon deadlines."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


@wp.kernel
def _record_horizon(tick: wp.array[int], horizons: wp.array[float], horizon: float):
    horizons[tick[0]] = horizon


@wp.kernel
def _increment_tick(tick: wp.array[int]):
    tick[0] += 1


@wp.kernel
def _advance_velocity(tick: wp.array[int], velocities: wp.array[float], body_qd: wp.array[wp.spatial_vector]):
    tick[0] += 1
    speed = velocities[tick[0] % velocities.shape[0]]
    body_qd[0] = wp.spatial_vector(speed, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_collision_schedule_horizon_deadline(test, device, external_capture=False):
    """Honor prediction deadlines despite rounded intervals and changing velocities."""
    if external_capture and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA graph capture requires conditional graph support")
    for substeps in (10, 12, 20):
        for profile in (
            "constant",
            "stationary",
            "fast",
            "slowing",
            "accelerating",
            "reversing",
            "varying",
            "overflow_then_rest",
            "accumulated_overflow",
        ):
            with test.subTest(substeps=substeps, profile=profile):
                speeds = np.full(substeps, 2.0, dtype=np.float32)
                if profile == "stationary":
                    speeds[:] = 0.0
                elif profile == "fast":
                    speeds[:] = 20.0
                elif profile == "slowing":
                    speeds[1:] = 0.02
                elif profile == "accelerating":
                    speeds[0] = 0.02
                elif profile == "reversing":
                    speeds[1::2] = -2.0
                elif profile == "varying":
                    speeds = np.random.default_rng(42).uniform(-5.0, 5.0, 2 * substeps).astype(np.float32)
                elif profile == "overflow_then_rest":
                    speeds = np.zeros(2 * substeps, dtype=np.float32)
                    speeds[0] = 20.0
                elif profile == "accumulated_overflow":
                    speeds = np.ones(substeps, dtype=np.float32)
                    speeds[4] = 8.0

                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                body = builder.add_body()
                builder.add_shape_sphere(body, radius=0.008)
                builder.body_qd[body] = (float(speeds[0]), 0.0, 0.0, 0.0, 0.0, 0.0)
                model = builder.finalize(device=device)
                states = (model.state(), model.state())
                pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.03)
                tick = wp.zeros(1, dtype=int, device=device)
                horizons = wp.zeros(2 * substeps, dtype=float, device=device)
                velocities = wp.array(speeds, dtype=float, device=device)
                dt = 1.0 / 60.0 / substeps

                def collide(state, horizon, *, tick=tick, horizons=horizons):
                    del state
                    wp.launch(_record_horizon, 1, inputs=[tick, horizons, horizon], device=device)

                def step(state_in, state_out, dt, *, tick=tick, velocities=velocities):
                    del dt
                    # Isolate scheduling from solver/contact behavior using prescribed velocities.
                    wp.copy(state_out.body_q, state_in.body_q)
                    wp.launch(_advance_velocity, 1, inputs=[tick, velocities, state_out.body_qd], device=device)

                scheduler = newton.CollisionSubstepScheduler(
                    pipeline,
                    states,
                    collision_callback=collide,
                    substep_callback=step,
                    frame_dt=1.0 / 60.0,
                    substeps=substeps,
                )
                if external_capture:
                    with wp.ScopedCapture(device=device) as capture:
                        scheduler.step()
                    for _ in range(2):
                        wp.capture_launch(capture.graph)
                else:
                    for _ in range(2):
                        scheduler.step()

                test.assertEqual(int(tick.numpy()[0]), 2 * substeps)
                recorded = horizons.numpy()
                # A scalar reference locks down refresh timing and horizon selection,
                # including frame resets and switching between short/long intervals.
                expected = np.zeros_like(recorded)
                intervals = [i for i in range(1, substeps + 1) if substeps % i == 0]
                expected_overflow = False
                for frame in range(2):
                    travel, previous_speed, deadline = 0.0, 0.0, 0
                    expected_overflow = False
                    for i in range(substeps):
                        index = frame * substeps + i
                        speed = abs(float(speeds[index % len(speeds)]))
                        relative_travel = 2.0 * speed * dt
                        if i > 0 and speed > previous_speed:
                            travel += 2.0 * (speed - previous_speed) * dt
                        accumulated_overflow = travel > 0.03
                        speed_increased = i > 0 and speed > previous_speed
                        if i == 0 or i >= deadline or speed_increased or travel + relative_travel > 0.03:
                            interval = max((j for j in intervals if j * relative_travel <= 0.03), default=1)
                            interval = min(interval, substeps - i)
                            expected[index] = interval * dt
                            deadline = i + interval
                            travel = 0.0
                        expected_overflow |= relative_travel > 0.03 or accumulated_overflow
                        travel += relative_travel
                        previous_speed = speed
                    final_speed = abs(float(speeds[((frame + 1) * substeps) % len(speeds)]))
                    if final_speed > previous_speed:
                        travel += 2.0 * (final_speed - previous_speed) * dt
                    expected_overflow |= travel > 0.03
                np.testing.assert_allclose(recorded, expected, rtol=1e-6, atol=0.0)
                events = np.flatnonzero(recorded)
                test.assertIn(0, events)
                test.assertIn(substeps, events)
                test.assertEqual(int(scheduler.interval_overflow.numpy()[0]), int(expected_overflow))
                for start, end in zip(events, np.append(events[1:], 2 * substeps), strict=True):
                    planned_steps = round(float(recorded[start]) / dt)
                    test.assertLessEqual(int(end - start), planned_steps, f"Expired horizon at substep {start}")
                    test.assertLessEqual(planned_steps, substeps - int(start) % substeps)
                if profile == "accumulated_overflow" and substeps == 10:
                    # Individual steps fit, but correcting prior travel exceeds the budget.
                    test.assertLess(2.0 * float(np.max(speeds)) * dt, 0.03)
                    test.assertEqual(int(scheduler.interval_overflow.numpy()[0]), 1)
                if profile == "accelerating":
                    test.assertLess(int(events[1]), substeps, "Retain travel-triggered early refreshes")
                if profile == "constant" and substeps == 10:
                    np.testing.assert_array_equal(events, np.arange(0, 2 * substeps, 2))
                elif profile == "fast":
                    np.testing.assert_array_equal(events, np.arange(2 * substeps))
                    np.testing.assert_allclose(recorded, dt, rtol=1e-6)
                elif profile == "stationary":
                    np.testing.assert_array_equal(events, [0, substeps])


def test_collision_schedule_refreshes_for_shape_acceleration(test, device):
    """Refresh a newly accelerating shape despite another shape's higher speed."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    slow_body = builder.add_body()
    builder.add_shape_sphere(slow_body, radius=0.1)
    fast_body = builder.add_body()
    builder.add_shape_sphere(fast_body, radius=0.1)
    builder.body_qd[fast_body] = (10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)
    states = (model.state(), model.state())
    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=10.0)
    tick = wp.zeros(1, dtype=int, device=device)
    horizons = wp.zeros(4, dtype=float, device=device)
    velocities = wp.array([0.0, 1.0, 1.0, 1.0], dtype=float, device=device)

    def collide(state, horizon):
        del state
        wp.launch(_record_horizon, 1, inputs=[tick, horizons, horizon], device=device)

    def step(state_in, state_out, dt):
        del dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.launch(_advance_velocity, 1, inputs=[tick, velocities, state_out.body_qd], device=device)

    scheduler = newton.CollisionSubstepScheduler(
        pipeline, states, collision_callback=collide, substep_callback=step, frame_dt=0.04, substeps=4
    )
    scheduler.step()
    np.testing.assert_array_equal(np.flatnonzero(horizons.numpy()), [0, 1])


def test_collision_schedule_checks_final_substep_overflow(test, device):
    """Report overflow caused by acceleration in the last solver substep."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    model = builder.finalize(device=device)
    states = (model.state(), model.state())
    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.5)
    tick = wp.zeros(1, dtype=int, device=device)
    velocities = wp.array([0.0, 0.0, 0.6], dtype=float, device=device)

    def collide(state, horizon):
        del state, horizon

    def step(state_in, state_out, dt):
        del dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.launch(_advance_velocity, 1, inputs=[tick, velocities, state_out.body_qd], device=device)

    scheduler = newton.CollisionSubstepScheduler(
        pipeline, states, collision_callback=collide, substep_callback=step, frame_dt=1.0, substeps=2
    )
    scheduler.step()
    test.assertEqual(int(scheduler.interval_overflow.numpy()[0]), 1)


def test_collision_schedule_preserves_integer_time_limits(test, device):
    """Accept time limits that round just below a whole substep count."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.1)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=1.0)

    for frame_dt, substeps, limit, expected_calls in (
        (3.0, 30, 0.3, 10),
        (1.0 / 60.0, 10, 1.0 / 600.0, 10),
    ):
        with test.subTest(frame_dt=frame_dt, limit=limit):
            tick = wp.zeros(1, dtype=int, device=device)

            def collide(state, horizon, *, tick=tick):
                del state, horizon
                wp.launch(_increment_tick, 1, inputs=[tick], device=device)

            def step(state_in, state_out, dt):
                del state_in, state_out, dt

            scheduler = newton.CollisionSubstepScheduler(
                pipeline,
                (model.state(), model.state()),
                collision_callback=collide,
                substep_callback=step,
                frame_dt=frame_dt,
                substeps=substeps,
                max_collision_dt=limit,
            )
            scheduler.step()
            test.assertEqual(int(tick.numpy()[0]), expected_calls)


@wp.kernel
def _measure_fingertip_penetration(body_q: wp.array[wp.transform], peak: wp.array[float]):
    center = wp.transform_point(body_q[0], wp.vec3(0.0, 0.08, 0.0))
    separation = wp.max(-center[0], center[0] - 0.001)
    peak[0] = wp.max(peak[0], 0.008 - separation)


def test_collision_schedule_rotating_fingertip(test, device):
    """Limit rotating fingertip overlap with a thin wall to solver-scale error."""
    for substeps, speed in ((10, 20.0), (20, 60.0)):
        with test.subTest(substeps=substeps, speed=speed):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            cfg = builder.ShapeConfig(gap=0.001, margin=0.0, mu=0.0)
            finger = builder.add_link()
            builder.add_shape_sphere(finger, radius=0.008, xform=wp.transform(wp.vec3(0.0, 0.08, 0.0)), cfg=cfg)
            joint = builder.add_joint_revolute(
                -1,
                finger,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(-0.04, -0.05, 0.0)),
                target_ke=0.0,
                target_kd=0.0,
                actuator_mode=newton.JointTargetMode.NONE,
                limit_lower=-10.0,
                limit_upper=10.0,
            )
            builder.add_articulation([joint])
            builder.joint_qd[0] = -speed
            builder.add_shape_box(
                -1, hx=0.0005, hy=0.15, hz=0.15, xform=wp.transform(wp.vec3(0.0005, 0.0, 0.0)), cfg=cfg
            )
            model = builder.finalize(device=device)
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)
            states = (model.state(), model.state())
            pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.03)
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverXPBD(model, iterations=30, rigid_contact_relaxation=0.8)
            peak = wp.zeros(1, dtype=float, device=device)

            def collide(state, horizon, *, pipeline=pipeline, contacts=contacts):
                pipeline.collide(state, contacts, dt=horizon)

            def step(state_in, state_out, dt, *, solver=solver, contacts=contacts, peak=peak):
                state_in.clear_forces()
                solver.step(state_in, state_out, None, contacts, dt)
                wp.launch(_measure_fingertip_penetration, 1, inputs=[state_out.body_q, peak], device=device)

            scheduler = newton.CollisionSubstepScheduler(
                pipeline,
                states,
                collision_callback=collide,
                substep_callback=step,
                frame_dt=1.0 / 60.0,
                substeps=substeps,
            )
            for _ in range(3):
                scheduler.step()
            test.assertLess(float(peak.numpy()[0]), 0.0001)


class TestCollisionSchedule(unittest.TestCase):
    """Test collision deadlines and their effect on thin-wall contact."""


add_function_test(
    TestCollisionSchedule,
    "test_collision_schedule_horizon_deadline",
    test_collision_schedule_horizon_deadline,
    devices=get_test_devices(),
)
add_function_test(
    TestCollisionSchedule,
    "test_collision_schedule_horizon_deadline_capture",
    test_collision_schedule_horizon_deadline,
    devices=get_cuda_test_devices(),
    external_capture=True,
)


add_function_test(
    TestCollisionSchedule,
    "test_collision_schedule_rotating_fingertip",
    test_collision_schedule_rotating_fingertip,
    devices=get_test_devices(),
)
add_function_test(
    TestCollisionSchedule,
    "test_collision_schedule_refreshes_for_shape_acceleration",
    test_collision_schedule_refreshes_for_shape_acceleration,
    devices=get_test_devices(),
)
add_function_test(
    TestCollisionSchedule,
    "test_collision_schedule_checks_final_substep_overflow",
    test_collision_schedule_checks_final_substep_overflow,
    devices=get_test_devices(),
)
add_function_test(
    TestCollisionSchedule,
    "test_collision_schedule_preserves_integer_time_limits",
    test_collision_schedule_preserves_integer_time_limits,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
