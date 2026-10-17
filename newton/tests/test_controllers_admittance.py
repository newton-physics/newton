# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`newton.controllers.ControllerAdmittance`."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton.controllers import ControllerAdmittance
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()

_IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _make_ctrl(
    device,
    *,
    robot_count=1,
    stiffness=100.0,
    damping=20.0,
    mass=1.0,
    operational_frame_pose_world=None,
    use_desired_wrench=False,
):
    kwargs = {}
    if operational_frame_pose_world is not None:
        kwargs["operational_frame_pose_world"] = operational_frame_pose_world
    return ControllerAdmittance(
        controlled_robot_count=robot_count,
        virtual_stiffness=stiffness,
        virtual_damping=damping,
        virtual_mass=mass,
        use_desired_wrench=use_desired_wrench,
        device=device,
        **kwargs,
    )


def _ports_at_rest(ctrl, robot_count=1):
    """Allocate ports with an identity reference pose, advancing the displacement in place."""
    inputs, outputs = ctrl.input(), ctrl.output()
    inputs.reference_tool_pose_operational.assign(np.tile(_IDENTITY_POSE, (robot_count, 1)))
    outputs.displacement_operational = inputs.displacement_operational
    outputs.displacement_twist_operational = inputs.displacement_twist_operational
    return inputs, outputs


def _run(ctrl, inputs, outputs, *, dt, steps):
    for _ in range(steps):
        ctrl.step(inputs=inputs, outputs=outputs, dt=dt)


def _virtual_energy(displacement, displacement_twist, stiffness, mass):
    return 0.5 * np.sum(mass * displacement_twist**2) + 0.5 * np.sum(stiffness * displacement**2)


def test_zero_wrench_at_rest_tracks_reference(test, device):
    """Report the reference pose and twist as the compliant ones when no wrench acts."""
    ctrl = _make_ctrl(device, operational_frame_pose_world=wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity()))
    inputs, outputs = _ports_at_rest(ctrl)
    reference_pose = np.array([0.1, -0.2, 0.3, 0.0, 0.0, np.sin(0.25), np.cos(0.25)], dtype=np.float32)
    reference_twist = np.array([0.5, 0.0, -0.1, 0.0, 0.2, 0.0], dtype=np.float32)
    inputs.reference_tool_pose_operational.assign(reference_pose[None])
    inputs.reference_twist_operational.assign(reference_twist[None])

    _run(ctrl, inputs, outputs, dt=0.01, steps=5)

    np.testing.assert_allclose(outputs.compliant_tool_pose_operational.numpy()[0], reference_pose, atol=1e-6)
    expected_world = reference_pose.copy()
    expected_world[:3] += [1.0, 2.0, 3.0]
    np.testing.assert_allclose(outputs.compliant_tool_pose_world.numpy()[0], expected_world, atol=1e-6)
    np.testing.assert_allclose(outputs.compliant_twist_operational.numpy()[0], reference_twist, atol=1e-6)
    np.testing.assert_allclose(inputs.displacement_operational.numpy(), 0.0, atol=1e-7)


def test_single_step_matches_backward_euler(test, device):
    """Advance every axis by exactly one backward-Euler step of M ë + D ė + K e = w_des - w_meas.

    Gains, displacement, and twist differ per axis, so an axis permutation
    would be caught. The rotation stays about one axis, where composing on
    the rotation group reduces to adding rotation vectors.
    """
    stiffness = np.array([100.0, 250.0, 40.0, 3.0, 5.0, 7.0])
    damping = np.array([10.0, 0.0, 30.0, 0.5, 0.2, 0.9])
    mass = np.array([2.0, 0.5, 0.0, 0.1, 0.3, 0.05])
    ctrl = _make_ctrl(
        device,
        stiffness=wp.spatial_vector(*stiffness),
        damping=wp.spatial_vector(*damping),
        mass=wp.spatial_vector(*mass),
        use_desired_wrench=True,
    )
    inputs, outputs = _ports_at_rest(ctrl)
    displacement = np.array([0.01, -0.02, 0.005, 0.0, 0.0, 0.3], dtype=np.float32)
    displacement_twist = np.array([0.1, 0.2, -0.3, 0.0, 0.0, -0.4], dtype=np.float32)
    measured = np.array([1.0, -2.0, 3.0, 0.0, 0.0, -0.5], dtype=np.float32)
    desired = np.array([0.5, 0.5, -1.0, 0.0, 0.0, 0.25], dtype=np.float32)
    inputs.displacement_operational.assign(displacement[None])
    inputs.displacement_twist_operational.assign(displacement_twist[None])
    inputs.measured_wrench_world.assign(measured[None])
    inputs.desired_wrench_world.assign(desired[None])
    dt = 0.01

    ctrl.step(inputs=inputs, outputs=outputs, dt=dt)

    wrench = desired - measured
    expected_twist = (mass * displacement_twist + dt * (wrench - stiffness * displacement)) / (
        mass + dt * damping + dt * dt * stiffness
    )
    expected_displacement = displacement + dt * expected_twist
    np.testing.assert_allclose(outputs.displacement_twist_operational.numpy()[0], expected_twist, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(outputs.displacement_operational.numpy()[0], expected_displacement, rtol=1e-5, atol=1e-6)


def test_static_wrench_settles_at_wrench_over_stiffness(test, device):
    """Settle each axis at e = -w_meas / K under a constant measured wrench, rotation included."""
    stiffness = np.array([100.0, 200.0, 400.0, 10.0, 20.0, 40.0])
    mass = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    damping = 2.0 * np.sqrt(stiffness * mass)  # critically damped
    ctrl = _make_ctrl(
        device,
        stiffness=wp.spatial_vector(*stiffness),
        damping=wp.spatial_vector(*damping),
        mass=wp.spatial_vector(*mass),
    )
    inputs, outputs = _ports_at_rest(ctrl)
    # The environment pushes back on the tool with -measured.
    measured = np.array([-5.0, 4.0, -8.0, 0.0, 0.0, -2.0], dtype=np.float32)
    inputs.measured_wrench_world.assign(measured[None])

    _run(ctrl, inputs, outputs, dt=0.005, steps=2000)

    np.testing.assert_allclose(inputs.displacement_operational.numpy()[0], -measured / stiffness, atol=1e-5)
    np.testing.assert_allclose(inputs.displacement_twist_operational.numpy()[0], 0.0, atol=1e-5)
    compliant_pose = outputs.compliant_tool_pose_operational.numpy()[0]
    np.testing.assert_allclose(compliant_pose[:3], (-measured / stiffness)[:3], atol=1e-5)
    half_angle = 0.5 * (-measured[5] / stiffness[5])
    np.testing.assert_allclose(compliant_pose[3:], [0.0, 0.0, np.sin(half_angle), np.cos(half_angle)], atol=1e-5)


def test_zero_stiffness_tracks_desired_force_on_unknown_surface(test, device):
    """Press until the contact force matches the setpoint, wherever the surface turns out to be.

    Closed loop against a stiff spring wall 20 mm below the reference, unknown
    to the controller: with zero virtual stiffness on z, the only steady state
    is measured == desired, reached 10 N / 5000 N/m = 2 mm into the wall.
    """
    wall_height = -0.02
    wall_stiffness = 5000.0
    desired_force = 10.0
    ctrl = _make_ctrl(
        device,
        stiffness=wp.spatial_vector(500.0, 500.0, 0.0, 50.0, 50.0, 50.0),
        damping=wp.spatial_vector(50.0, 50.0, 400.0, 5.0, 5.0, 5.0),
        mass=wp.spatial_vector(1.0, 1.0, 1.0, 0.1, 0.1, 0.1),
        use_desired_wrench=True,
    )
    inputs, outputs = _ports_at_rest(ctrl)
    # Pressing down: the tool exerts -z force on the wall.
    inputs.desired_wrench_world.assign(np.array([[0.0, 0.0, -desired_force, 0.0, 0.0, 0.0]], dtype=np.float32))

    measured = np.zeros((1, 6), dtype=np.float32)
    for _ in range(1500):
        inputs.measured_wrench_world.assign(measured)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.002)
        tool_height = outputs.compliant_tool_pose_world.numpy()[0, 2]
        measured[0, 2] = -wall_stiffness * max(0.0, wall_height - tool_height)

    test.assertAlmostEqual(-measured[0, 2], desired_force, delta=0.05)
    test.assertAlmostEqual(float(outputs.compliant_tool_pose_world.numpy()[0, 2]), wall_height - 0.002, delta=1e-5)


def test_energy_never_increases_for_stiff_gains_and_large_dt(test, device):
    """Dissipate virtual energy every step even where explicit integration would diverge.

    With K / M = 1e8 and dt = 0.01, sqrt(K / M) * dt = 100, far past the
    stability bound of 2 that explicit and semi-implicit Euler share.
    """
    stiffness = np.full(6, 1.0e6)
    mass = np.full(6, 1.0e-2)
    ctrl = _make_ctrl(device, stiffness=1.0e6, damping=0.0, mass=1.0e-2)
    inputs, outputs = _ports_at_rest(ctrl)
    inputs.displacement_operational.assign(np.array([[0.01, -0.02, 0.03, 0.1, -0.2, 0.3]], dtype=np.float32))
    inputs.displacement_twist_operational.assign(np.array([[1.0, 2.0, -1.0, 0.5, 0.5, -0.5]], dtype=np.float32))

    energy = _virtual_energy(
        inputs.displacement_operational.numpy()[0], inputs.displacement_twist_operational.numpy()[0], stiffness, mass
    )
    for _ in range(50):
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        next_energy = _virtual_energy(
            inputs.displacement_operational.numpy()[0],
            inputs.displacement_twist_operational.numpy()[0],
            stiffness,
            mass,
        )
        test.assertTrue(np.isfinite(next_energy))
        test.assertLessEqual(next_energy, energy * (1.0 + 1e-4) + 1e-9)
        energy = next_energy


def test_operational_frame_rotates_wrench_and_composes_world_pose(test, device):
    """Read gains in the operational frame, and express the world-frame wrench there first.

    The operational frame is rotated +90 degrees about world z, so world +x
    is operational -y. A wrench along world x must deflect the operational
    y axis by its own stiffness, and the world pose must deflect along x.
    """
    frame = wp.transform(wp.vec3(0.5, 0.0, 0.2), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi))
    ctrl = _make_ctrl(
        device,
        stiffness=wp.spatial_vector(1000.0, 100.0, 1000.0, 10.0, 10.0, 10.0),
        damping=wp.spatial_vector(400.0, 40.0, 400.0, 4.0, 4.0, 4.0),
        mass=0.0,
        operational_frame_pose_world=frame,
    )
    inputs, outputs = _ports_at_rest(ctrl)
    # The environment pushes the tool along world +x with 2 N.
    inputs.measured_wrench_world.assign(np.array([[-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    _run(ctrl, inputs, outputs, dt=0.01, steps=1000)

    np.testing.assert_allclose(inputs.displacement_operational.numpy()[0], [0.0, -0.02, 0.0, 0.0, 0.0, 0.0], atol=1e-5)
    world_pose = outputs.compliant_tool_pose_world.numpy()[0]
    np.testing.assert_allclose(world_pose[:3], [0.52, 0.0, 0.2], atol=1e-5)


def test_compliant_twist_rotates_reference_angular_twist_by_offset(test, device):
    """Report the angular twist of q_offset * q_reference, not the sum of the two twists."""
    ctrl = _make_ctrl(device, stiffness=1.0, damping=0.0, mass=1.0e6)
    inputs, outputs = _ports_at_rest(ctrl)
    offset_angle = 0.5 * np.pi
    inputs.displacement_operational.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, offset_angle]], dtype=np.float32))
    inputs.reference_twist_operational.assign(np.array([[0.3, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32))

    ctrl.step(inputs=inputs, outputs=outputs, dt=1.0e-4)

    # The offset (~90 degrees about z) turns the reference's x angular rate into y.
    np.testing.assert_allclose(
        outputs.compliant_twist_operational.numpy()[0], [0.3, 0.0, 0.0, 0.0, 1.0, 0.0], atol=1e-3
    )


def test_live_gains_change_the_equilibrium(test, device):
    """Read live gains every step, so switching stiffness mid-run moves the equilibrium."""
    ctrl = _make_ctrl(device, stiffness=None, damping=None, mass=None)
    inputs, outputs = _ports_at_rest(ctrl)
    inputs.virtual_damping.fill_(wp.spatial_vector(60.0, 60.0, 60.0, 6.0, 6.0, 6.0))
    inputs.virtual_mass.fill_(wp.spatial_vector(1.0, 1.0, 1.0, 0.1, 0.1, 0.1))
    inputs.measured_wrench_world.assign(np.array([[-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    inputs.virtual_stiffness.fill_(wp.spatial_vector(1000.0, 1000.0, 1000.0, 10.0, 10.0, 10.0))
    _run(ctrl, inputs, outputs, dt=0.005, steps=1000)
    test.assertAlmostEqual(float(inputs.displacement_operational.numpy()[0, 0]), 0.01, delta=1e-5)

    inputs.virtual_stiffness.fill_(wp.spatial_vector(250.0, 1000.0, 1000.0, 10.0, 10.0, 10.0))
    _run(ctrl, inputs, outputs, dt=0.005, steps=1000)
    test.assertAlmostEqual(float(inputs.displacement_operational.numpy()[0, 0]), 0.04, delta=1e-5)


def test_live_axis_without_positive_gain_is_held_still(test, device):
    """Hold a live axis with no positive gain still instead of dividing by zero.

    The rotational axis is held at zero rate, but composing the other two
    axes' rotations can still leak a second-order amount into its rotation
    vector, so only the linear axis is exactly zero.
    """
    ctrl = _make_ctrl(device, stiffness=None, damping=None, mass=None)
    inputs, outputs = _ports_at_rest(ctrl)
    inputs.virtual_stiffness.fill_(wp.spatial_vector(100.0, 0.0, 100.0, 1.0, 1.0, -1.0))
    inputs.virtual_damping.fill_(wp.spatial_vector(10.0, 0.0, 10.0, 1.0, 1.0, 0.0))
    inputs.virtual_mass.fill_(wp.spatial_vector(1.0, 0.0, 1.0, 1.0, 1.0, 0.0))
    inputs.measured_wrench_world.assign(np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32))

    _run(ctrl, inputs, outputs, dt=0.01, steps=10)

    displacement = inputs.displacement_operational.numpy()[0]
    test.assertTrue(np.all(np.isfinite(displacement)))
    test.assertEqual(float(displacement[1]), 0.0)
    test.assertEqual(float(inputs.displacement_twist_operational.numpy()[0, 5]), 0.0)
    test.assertLess(abs(float(displacement[5])), 1e-6)
    test.assertNotEqual(float(displacement[0]), 0.0)


def test_robots_are_independent(test, device):
    """Apply each robot's own gains and wrench, with no cross-talk between slots."""
    stiffness = wp.array(
        [wp.spatial_vector(100.0, 100.0, 100.0, 1.0, 1.0, 1.0), wp.spatial_vector(400.0, 400.0, 400.0, 4.0, 4.0, 4.0)],
        dtype=wp.spatial_vector,
        device=device,
    )
    ctrl = _make_ctrl(device, robot_count=2, stiffness=stiffness, damping=20.0, mass=0.0)
    inputs, outputs = _ports_at_rest(ctrl, robot_count=2)
    inputs.measured_wrench_world.assign(
        np.array([[-4.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -4.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    )

    _run(ctrl, inputs, outputs, dt=0.01, steps=500)

    np.testing.assert_allclose(
        inputs.displacement_operational.numpy(),
        [[0.04, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.0, 0.0, 0.0]],
        atol=1e-5,
    )


def test_rotation_displacement_stays_within_pi(test, device):
    """Keep the rotation vector at an angle of at most pi under a moment too large for the spring."""
    ctrl = _make_ctrl(device, stiffness=0.1, damping=1.0, mass=0.01)
    inputs, outputs = _ports_at_rest(ctrl)
    inputs.measured_wrench_world.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -5.0]], dtype=np.float32))

    for _ in range(300):
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        angle = np.linalg.norm(inputs.displacement_operational.numpy()[0, 3:])
        test.assertTrue(np.isfinite(angle))
        test.assertLessEqual(angle, np.pi + 1e-5)


def test_in_place_matches_ping_pong(test, device):
    """Advance identically whether the displacement outputs alias their inputs or not."""
    measured = np.array([[1.0, -2.0, 0.5, 0.1, -0.2, 0.3]], dtype=np.float32)

    in_place = _make_ctrl(device)
    in_place_inputs, in_place_outputs = _ports_at_rest(in_place)
    in_place_inputs.measured_wrench_world.assign(measured)

    ping_pong = _make_ctrl(device)
    ping_pong_inputs, ping_pong_outputs = ping_pong.input(), ping_pong.output()
    ping_pong_inputs.reference_tool_pose_operational.assign(_IDENTITY_POSE[None])
    ping_pong_inputs.measured_wrench_world.assign(measured)

    for _ in range(20):
        in_place.step(inputs=in_place_inputs, outputs=in_place_outputs, dt=0.01)
        ping_pong.step(inputs=ping_pong_inputs, outputs=ping_pong_outputs, dt=0.01)
        wp.copy(ping_pong_inputs.displacement_operational, ping_pong_outputs.displacement_operational)
        wp.copy(ping_pong_inputs.displacement_twist_operational, ping_pong_outputs.displacement_twist_operational)

    np.testing.assert_array_equal(
        in_place_outputs.displacement_operational.numpy(), ping_pong_outputs.displacement_operational.numpy()
    )
    np.testing.assert_array_equal(
        in_place_outputs.compliant_tool_pose_world.numpy(), ping_pong_outputs.compliant_tool_pose_world.numpy()
    )


def test_indexed_view_ports_match_plain_arrays(test, device):
    """Gather inputs from and scatter outputs to views of larger arrays, matching plain ports."""
    measured = np.array([[-3.0, 1.0, 0.0, 0.0, 0.2, 0.0], [0.0, 0.0, -6.0, 0.1, 0.0, 0.0]], dtype=np.float32)

    plain = _make_ctrl(device, robot_count=2)
    plain_inputs, plain_outputs = _ports_at_rest(plain, robot_count=2)
    plain_inputs.measured_wrench_world.assign(measured)

    # Robots live in slots 3 and 1 of five-slot simulation-sized arrays.
    slots = wp.array([3, 1], dtype=wp.int32, device=device)
    viewed = _make_ctrl(device, robot_count=2)
    viewed_inputs, viewed_outputs = viewed.input(), viewed.output()
    sim_reference = wp.array(np.tile(_IDENTITY_POSE, (5, 1)), dtype=wp.transform, device=device)
    sim_measured = np.zeros((5, 6), dtype=np.float32)
    sim_measured[[3, 1]] = measured
    sim_displacement = wp.zeros(5, dtype=wp.spatial_vector, device=device)
    sim_displacement_twist = wp.zeros(5, dtype=wp.spatial_vector, device=device)
    sim_pose_world = wp.zeros(5, dtype=wp.transform, device=device)
    viewed_inputs.reference_tool_pose_operational = sim_reference[slots]
    viewed_inputs.measured_wrench_world = wp.array(sim_measured, dtype=wp.spatial_vector, device=device)[slots]
    viewed_inputs.displacement_operational = sim_displacement[slots]
    viewed_inputs.displacement_twist_operational = sim_displacement_twist[slots]
    viewed_outputs.displacement_operational = sim_displacement[slots]
    viewed_outputs.displacement_twist_operational = sim_displacement_twist[slots]
    viewed_outputs.compliant_tool_pose_world = sim_pose_world[slots]

    for _ in range(10):
        plain.step(inputs=plain_inputs, outputs=plain_outputs, dt=0.01)
        viewed.step(inputs=viewed_inputs, outputs=viewed_outputs, dt=0.01)

    np.testing.assert_allclose(
        sim_displacement.numpy()[[3, 1]], plain_outputs.displacement_operational.numpy(), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        sim_pose_world.numpy()[[3, 1]], plain_outputs.compliant_tool_pose_world.numpy(), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(sim_displacement.numpy()[[0, 2, 4]], 0.0)


def test_graph_replay_matches_eager_steps(test, device):
    """Capture one step in a CUDA graph and replay it, with dt read from an array."""
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        test.skipTest("graph capture needs a CUDA device with mempool enabled")
    measured = np.array([[-1.0, 2.0, -3.0, 0.1, 0.2, -0.3]], dtype=np.float32)

    eager = _make_ctrl(device, stiffness=None)
    eager_inputs, eager_outputs = _ports_at_rest(eager)
    eager_inputs.measured_wrench_world.assign(measured)
    eager_inputs.virtual_stiffness.fill_(wp.spatial_vector(300.0, 300.0, 300.0, 3.0, 3.0, 3.0))

    graphed = _make_ctrl(device, stiffness=None)
    graphed_inputs, graphed_outputs = _ports_at_rest(graphed)
    graphed_inputs.measured_wrench_world.assign(measured)
    graphed_inputs.virtual_stiffness.fill_(wp.spatial_vector(300.0, 300.0, 300.0, 3.0, 3.0, 3.0))
    dt = wp.array([0.004], dtype=wp.float32, device=device)

    with wp.ScopedDevice(device):
        graphed.step(inputs=graphed_inputs, outputs=graphed_outputs, dt=dt)  # compile outside the capture
        graphed_inputs.displacement_operational.zero_()
        graphed_inputs.displacement_twist_operational.zero_()
        with wp.ScopedCapture() as capture:
            graphed.step(inputs=graphed_inputs, outputs=graphed_outputs, dt=dt)

    for step_dt in (0.004, 0.004, 0.002, 0.008):
        dt.fill_(step_dt)
        wp.capture_launch(capture.graph)
        eager.step(inputs=eager_inputs, outputs=eager_outputs, dt=step_dt)

    np.testing.assert_allclose(
        graphed_outputs.displacement_operational.numpy(), eager_outputs.displacement_operational.numpy(), atol=1e-7
    )
    np.testing.assert_allclose(
        graphed_outputs.compliant_tool_pose_world.numpy(), eager_outputs.compliant_tool_pose_world.numpy(), atol=1e-7
    )


class TestControllerAdmittance(unittest.TestCase):
    def test_negative_baked_gain_raises(self):
        """Reject a baked gain with a negative axis."""
        with self.assertRaises(ValueError):
            _make_ctrl(None, damping=wp.spatial_vector(1.0, 1.0, -1.0, 1.0, 1.0, 1.0))

    def test_baked_axis_without_positive_gain_raises(self):
        """Reject fully baked gains that leave an axis with no positive gain."""
        with self.assertRaisesRegex(ValueError, r"axes \[2\]"):
            _make_ctrl(
                None,
                stiffness=wp.spatial_vector(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
                damping=wp.spatial_vector(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
                mass=0.0,
            )

    def test_bool_robot_count_raises(self):
        """Reject a bool robot count rather than reading it as 1."""
        with self.assertRaises(TypeError):
            _make_ctrl(None, robot_count=True)

    def test_requires_grad_raises(self):
        """Reject requires_grad, which the controller does not support yet."""
        with self.assertRaises(ValueError):
            ControllerAdmittance(
                controlled_robot_count=1,
                virtual_stiffness=1.0,
                virtual_damping=1.0,
                virtual_mass=1.0,
                requires_grad=True,
            )

    def test_input_for_disabled_feature_raises(self):
        """Reject a port that a baked gain or disabled feature would silently ignore."""
        ctrl = _make_ctrl(None)
        inputs, outputs = _ports_at_rest(ctrl)
        inputs.desired_wrench_world = wp.zeros(1, dtype=wp.spatial_vector)
        with self.assertRaisesRegex(ValueError, "use_desired_wrench"):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        inputs.desired_wrench_world = None
        inputs.virtual_stiffness = wp.zeros(1, dtype=wp.spatial_vector)
        with self.assertRaisesRegex(ValueError, "virtual_stiffness"):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

    def test_non_positive_dt_raises_before_writing_outputs(self):
        """Reject dt <= 0 without touching the (aliased) displacement."""
        ctrl = _make_ctrl(None)
        inputs, outputs = _ports_at_rest(ctrl)
        inputs.displacement_operational.assign(np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        inputs.measured_wrench_world.assign(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        for dt in (0.0, -0.01):
            with self.assertRaises(ValueError):
                ctrl.step(inputs=inputs, outputs=outputs, dt=dt)
        np.testing.assert_array_equal(
            inputs.displacement_operational.numpy()[0], np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

    def test_wrong_shape_port_raises(self):
        """Reject a port whose length differs from controlled_robot_count."""
        ctrl = _make_ctrl(None, robot_count=2)
        inputs, outputs = _ports_at_rest(ctrl, robot_count=2)
        inputs.measured_wrench_world = wp.zeros(3, dtype=wp.spatial_vector)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

    def test_wrong_dtype_output_raises(self):
        """Reject an output bound to an array of the wrong dtype."""
        ctrl = _make_ctrl(None)
        inputs, outputs = _ports_at_rest(ctrl)
        outputs.compliant_tool_pose_world = wp.zeros(1, dtype=wp.spatial_vector)
        with self.assertRaises(TypeError):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)


add_function_test(
    TestControllerAdmittance,
    "test_zero_wrench_at_rest_tracks_reference",
    test_zero_wrench_at_rest_tracks_reference,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_single_step_matches_backward_euler",
    test_single_step_matches_backward_euler,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_static_wrench_settles_at_wrench_over_stiffness",
    test_static_wrench_settles_at_wrench_over_stiffness,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_zero_stiffness_tracks_desired_force_on_unknown_surface",
    test_zero_stiffness_tracks_desired_force_on_unknown_surface,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_energy_never_increases_for_stiff_gains_and_large_dt",
    test_energy_never_increases_for_stiff_gains_and_large_dt,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_operational_frame_rotates_wrench_and_composes_world_pose",
    test_operational_frame_rotates_wrench_and_composes_world_pose,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_compliant_twist_rotates_reference_angular_twist_by_offset",
    test_compliant_twist_rotates_reference_angular_twist_by_offset,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_live_gains_change_the_equilibrium",
    test_live_gains_change_the_equilibrium,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_live_axis_without_positive_gain_is_held_still",
    test_live_axis_without_positive_gain_is_held_still,
    devices=devices,
)
add_function_test(TestControllerAdmittance, "test_robots_are_independent", test_robots_are_independent, devices=devices)
add_function_test(
    TestControllerAdmittance,
    "test_rotation_displacement_stays_within_pi",
    test_rotation_displacement_stays_within_pi,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance, "test_in_place_matches_ping_pong", test_in_place_matches_ping_pong, devices=devices
)
add_function_test(
    TestControllerAdmittance,
    "test_indexed_view_ports_match_plain_arrays",
    test_indexed_view_ports_match_plain_arrays,
    devices=devices,
)
add_function_test(
    TestControllerAdmittance,
    "test_graph_replay_matches_eager_steps",
    test_graph_replay_matches_eager_steps,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
