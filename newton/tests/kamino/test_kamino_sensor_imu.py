# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test Kamino body acceleration and SensorIMU integration."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorIMU
from newton.tests.unittest_utils import add_function_test, get_test_devices


DT = 1.0 / 120.0
INERTIA = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _add_free_body(builder, *, label, xform=None):
    """Add a unit free body and return its body index."""
    return builder.add_body(
        label=label,
        xform=wp.transform_identity() if xform is None else xform,
        mass=1.0,
        inertia=INERTIA,
        lock_inertia=True,
    )


def _make_free_body_scene(
    device,
    *,
    gravity=(0.0, 0.0, -9.81),
    site_xform=None,
    enable_contacts=False,
):
    """Build one free body with an IMU site."""
    builder = newton.ModelBuilder()
    builder.begin_world(gravity=gravity)
    body_xform = wp.transform(wp.vec3(0.0, 0.0, 10.0), wp.quat_identity()) if enable_contacts else None
    body = _add_free_body(builder, label="body", xform=body_xform)
    site = builder.add_site(body, label="imu", xform=site_xform)
    if enable_contacts:
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_ground_plane()
    builder.end_world()
    model = builder.finalize(device=device)
    sensor = SensorIMU(model, sites=[site])
    return model, body, sensor


def _make_solver(model, integrator="euler"):
    """Create a Kamino solver configured for an integration scheme."""
    config = newton.solvers.SolverKamino.Config(
        integrator=integrator,
        use_collision_detector=integrator == "moreau",
    )
    return newton.solvers.SolverKamino(model, config=config)


def test_free_fall_body_acceleration_and_imu(test, device, integrator):
    """Report gravity as body acceleration and zero free-fall specific force."""
    model, _, sensor = _make_free_body_scene(device, enable_contacts=integrator == "moreau")
    solver = _make_solver(model, integrator)
    state_in = model.state()
    state_out = model.state()

    solver.step(state_in, state_out, model.control(), None, DT)
    sensor.update(state_out)

    expected_gravity = np.array([0.0, 0.0, -9.81])
    np.testing.assert_allclose(state_out.body_qdd.numpy()[0, :3], expected_gravity, rtol=0.0, atol=2.0e-4)
    np.testing.assert_allclose(state_out.body_qdd.numpy()[0, 3:], 0.0, rtol=0.0, atol=2.0e-4)
    np.testing.assert_allclose(sensor.accelerometer.numpy()[0], 0.0, rtol=0.0, atol=2.0e-4)
    np.testing.assert_allclose(sensor.gyroscope.numpy()[0], 0.0, rtol=0.0, atol=2.0e-4)


def test_body_acceleration_state_ownership(test, device):
    """Preserve input acceleration and support ping-pong and in-place stepping."""
    model, _, _ = _make_free_body_scene(device)
    solver = _make_solver(model)
    control = model.control()
    state_a = model.state()
    state_b = model.state()
    state_a.body_qdd.fill_(17.0)

    velocity_a = state_a.body_qd.numpy().copy()
    solver.step(state_a, state_b, control, None, DT)
    np.testing.assert_array_equal(state_a.body_qdd.numpy(), 17.0)
    np.testing.assert_allclose(
        state_b.body_qdd.numpy(),
        (state_b.body_qd.numpy() - velocity_a) / DT,
        rtol=0.0,
        atol=2.0e-5,
    )

    velocity_b = state_b.body_qd.numpy().copy()
    solver.step(state_b, state_a, control, None, DT)
    np.testing.assert_allclose(
        state_a.body_qdd.numpy(),
        (state_a.body_qd.numpy() - velocity_b) / DT,
        rtol=0.0,
        atol=2.0e-5,
    )

    state_in_place = model.state()
    velocity_in_place = state_in_place.body_qd.numpy().copy()
    solver.step(state_in_place, state_in_place, control, None, DT)
    np.testing.assert_allclose(
        state_in_place.body_qdd.numpy(),
        (state_in_place.body_qd.numpy() - velocity_in_place) / DT,
        rtol=0.0,
        atol=2.0e-5,
    )


def test_rotating_offset_imu(test, device):
    """Include angular acceleration and centripetal acceleration at an offset site."""
    radius = 0.5
    site_xform = wp.transform(wp.vec3(radius, 0.0, 0.0), wp.quat_identity())
    model, _, sensor = _make_free_body_scene(device, gravity=(0.0, 0.0, 0.0), site_xform=site_xform)
    solver = _make_solver(model)
    state_in = model.state()
    state_out = model.state()
    state_in.body_qd.assign([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
    state_in.body_f.assign([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    solver.step(state_in, state_out, model.control(), None, DT)
    sensor.update(state_out)

    angular_velocity = 2.0 + DT
    np.testing.assert_allclose(state_out.body_qdd.numpy()[0, 3:], [0.0, 0.0, 1.0], rtol=0.0, atol=3.0e-4)
    np.testing.assert_allclose(
        sensor.accelerometer.numpy()[0],
        [-angular_velocity * angular_velocity * radius, radius, 0.0],
        rtol=0.0,
        atol=5.0e-4,
    )
    np.testing.assert_allclose(sensor.gyroscope.numpy()[0], [0.0, 0.0, angular_velocity], rtol=0.0, atol=3.0e-4)


def test_supported_body_imu(test, device):
    """Report an upward specific force after a body settles on a plane."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    body = _add_free_body(
        builder,
        label="box",
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()),
    )
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    site = builder.add_site(body, label="imu")
    builder.add_ground_plane(cfg=cfg)
    builder.end_world()
    model = builder.finalize(device=device)
    sensor = SensorIMU(model, sites=[site])
    solver = newton.solvers.SolverKamino(
        model,
        config=newton.solvers.SolverKamino.Config(use_collision_detector=True),
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    saw_impact_acceleration = False

    for _ in range(180):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, DT)
        saw_impact_acceleration |= np.linalg.norm(state_out.body_qdd.numpy()[body, :3]) > 20.0
        state_in, state_out = state_out, state_in

    sensor.update(state_in)

    test.assertTrue(saw_impact_acceleration)
    np.testing.assert_allclose(state_in.body_qd.numpy()[body], 0.0, rtol=0.0, atol=2.0e-2)
    np.testing.assert_allclose(sensor.accelerometer.numpy()[0], [0.0, 0.0, 9.81], rtol=0.0, atol=3.0e-1)


def test_body_acceleration_partial_reset(test, device):
    """Clear acceleration only in reset worlds and clear all worlds on full reset."""
    builder = newton.ModelBuilder()
    for world_index in range(2):
        builder.begin_world(label=f"world_{world_index}")
        _add_free_body(builder, label=f"body_{world_index}")
        builder.end_world()
    builder.request_state_attributes("body_qdd")
    model = builder.finalize(device=device)
    solver = _make_solver(model)
    state = model.state()
    state.body_qdd.assign(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]
    )
    world_mask = wp.array([True, False, False], dtype=wp.bool, device=device)

    solver.reset(state, world_mask=world_mask, flags=0)
    acceleration = state.body_qdd.numpy()
    np.testing.assert_array_equal(acceleration[0], 0.0)
    np.testing.assert_array_equal(acceleration[1], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    solver.reset(state, flags=0)
    np.testing.assert_array_equal(state.body_qdd.numpy(), 0.0)


def test_body_acceleration_remains_unrequested(test, device):
    """Leave the optional acceleration absent when it is not requested."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    _add_free_body(builder, label="body")
    builder.end_world()
    model = builder.finalize(device=device)
    solver = _make_solver(model)
    state = model.state()

    test.assertIsNone(state.body_qdd)

    solver.step(state, state, model.control(), None, DT)
    test.assertIsNone(state.body_qdd)


def test_body_acceleration_cuda_graph(test, device):
    """Replay captured in-place steps with valid body acceleration."""
    if not device.is_cuda or wp.config.verify_cuda:
        test.skipTest("CUDA graph capture requires CUDA without verification mode")

    model, _, _ = _make_free_body_scene(device)
    solver = _make_solver(model)
    state = model.state()
    control = model.control()

    solver.step(state, state, control, None, DT)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, state, control, None, DT)
    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)

    np.testing.assert_allclose(state.body_qdd.numpy()[0, :3], [0.0, 0.0, -9.81], rtol=0.0, atol=2.0e-4)


class TestKaminoSensorIMU(unittest.TestCase):
    """Test Kamino body acceleration support."""


devices = get_test_devices(mode="basic")

for _integrator in ("euler", "moreau"):
    add_function_test(
        TestKaminoSensorIMU,
        f"test_free_fall_body_acceleration_and_imu_{_integrator}",
        test_free_fall_body_acceleration_and_imu,
        devices=devices,
        integrator=_integrator,
        check_output=False,
    )

add_function_test(
    TestKaminoSensorIMU,
    "test_body_acceleration_state_ownership",
    test_body_acceleration_state_ownership,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestKaminoSensorIMU,
    "test_rotating_offset_imu",
    test_rotating_offset_imu,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestKaminoSensorIMU,
    "test_supported_body_imu",
    test_supported_body_imu,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestKaminoSensorIMU,
    "test_body_acceleration_partial_reset",
    test_body_acceleration_partial_reset,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestKaminoSensorIMU,
    "test_body_acceleration_remains_unrequested",
    test_body_acceleration_remains_unrequested,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestKaminoSensorIMU,
    "test_body_acceleration_cuda_graph",
    test_body_acceleration_cuda_graph,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
