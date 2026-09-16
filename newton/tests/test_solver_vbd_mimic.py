# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _advance_reference(
    angles: wp.array[float],
    increment: float,
    flip_sign: bool,
    body_q: wp.array[wp.transform],
):
    pair = wp.tid()
    angle = angles[pair] + increment
    angles[pair] = angle
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
    if flip_sign:
        rotation = -rotation
    body_q[2 * pair] = wp.transform(wp.vec3(0.0), rotation)


@wp.kernel
def _prescribe_d6_reference(linear: float, angular: float, body_q: wp.array[wp.transform]):
    body_q[1] = wp.transform(wp.vec3(0.0, 0.0, linear), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angular))


class _MimicRig:
    def __init__(
        self,
        device,
        *,
        ratio=0.01,
        offset=0.03,
        initial=0.0,
        angular_follower=False,
        d6=False,
        worlds=False,
        kinematic=True,
    ):
        self.ratio, self.offset = ratio, offset
        self.angular_follower = angular_follower
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for world in (0, 1, -1) if worlds else (-1,):
            if world >= 0:
                builder.begin_world()
            parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True, is_kinematic=kinematic)
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
            cfg = newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.Z, target_ke=0.0, target_kd=0.0, limit_ke=0.0, limit_kd=0.0
            )
            if d6:
                reference = builder.add_joint_d6(-1, parent, angular_axes=[cfg])
            else:
                reference = builder.add_joint(newton.JointType.REVOLUTE, -1, parent, angular_axes=[cfg])
            if angular_follower:
                follower = builder.add_joint(newton.JointType.REVOLUTE, -1, child, angular_axes=[cfg])
            else:
                follower = builder.add_joint(newton.JointType.PRISMATIC, -1, child, linear_axes=[cfg])
            builder.add_articulation([reference, follower])
            builder.set_joint_mimic(follower, reference, (offset, ratio))
            builder.joint_q[-2:] = [initial, offset + ratio * initial]
            if world >= 0:
                builder.end_world()
        builder.color()
        self.model = builder.finalize(device=device)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.state, self.next_state = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=12, rigid_compliant_alm=True)
        self.angles = wp.full(3 if worlds else 1, initial, dtype=float, device=device)

    def step(self, increment=0.1, flip_sign=False):
        """Prescribe the reference pose and solve the follower without setting its coordinate."""
        wp.launch(
            _advance_reference,
            dim=self.angles.size,
            inputs=[self.angles, increment, flip_sign],
            outputs=[self.state.body_q],
            device=self.model.device,
        )
        self.solver.step(self.state, self.next_state, self.control, None, 1.0 / 240.0)
        self.state, self.next_state = self.next_state, self.state

    def check(self):
        """Compare actual follower poses against the commanded, accumulated reference angles."""
        targets = self.offset + self.ratio * self.angles.numpy()
        poses = self.state.body_q.numpy()[1::2]
        if self.angular_follower:
            for pose, angle in zip(poses, targets, strict=True):
                expected = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(angle)))
                np.testing.assert_allclose(abs(np.dot(pose[3:], expected)), 1.0, atol=2.0e-6)
        else:
            np.testing.assert_allclose(poses[:, 2], targets, atol=2.0e-5)


def test_vbd_mimic_multiturn_translation(test, device):
    """Extend a lead screw continuously over several turns in either direction."""
    for d6 in (False, True):
        for ratio in (0.01, -0.02):
            with test.subTest(d6=d6, ratio=ratio):
                rig = _MimicRig(device, ratio=ratio, d6=d6)
                for direction in (1.0, -1.0, -1.0):
                    for step in range(100):
                        rig.step(direction * 0.1, flip_sign=step % 2 == 0)
                        rig.check()


def test_vbd_mimic_fractional_angular_ratio(test, device):
    """Preserve fractional gear ratios when either angular joint crosses a turn."""
    for ratio in (0.5, -1.5):
        rig = _MimicRig(device, ratio=ratio, angular_follower=True)
        for _ in range(180):
            rig.step()
            rig.check()


def test_vbd_mimic_multiturn_dynamics(test, device):
    """Keep a driven lead screw coupled across turns with friction and damping on both joints."""
    rig = _MimicRig(device, kinematic=False)
    rig.model.joint_friction.assign([0.2, 2.0])
    rig.model.joint_damping.assign([0.2, 20.0])
    speed = 12.0
    rig.state.joint_qd.assign([speed, rig.ratio * speed])
    newton.eval_fk(rig.model, rig.state.joint_q, rig.state.joint_qd, rig.state)
    # Balance both joints' dissipation to maintain approximately constant speed.
    torque = 0.2 + rig.ratio * 2.0 + (0.2 + rig.ratio**2 * 20.0) * speed
    rig.control.joint_f.assign([torque, 0.0])
    angle, previous = 0.0, 0.0
    for _ in range(300):
        rig.solver.step(rig.state, rig.next_state, rig.control, None, 1.0 / 240.0)
        rig.state, rig.next_state = rig.next_state, rig.state
        poses = rig.state.body_q.numpy()
        wrapped = 2.0 * math.atan2(poses[0, 5], poses[0, 6])
        angle += math.remainder(wrapped - previous, 2.0 * math.pi)
        previous = wrapped
        test.assertAlmostEqual(float(poses[1, 2]), rig.offset + rig.ratio * angle, delta=2.0e-5)
    test.assertGreater(angle, 4.0 * math.pi)
    newton.eval_ik(rig.model, rig.state, rig.state.joint_q, rig.state.joint_qd)
    velocities = rig.state.joint_qd.numpy()
    np.testing.assert_allclose(velocities[1], rig.ratio * velocities[0], atol=2.0e-5)


def test_vbd_mimic_initial_turns(test, device):
    """Seed turns from authored joint coordinates, including non-model initial states."""
    for initial in (-4.0 * math.pi - 0.2, 6.0 * math.pi + 0.2):
        rig = _MimicRig(device, initial=initial)
        rig.step(0.0, flip_sign=True)
        rig.check()
        rig = _MimicRig(device)
        rig.state.joint_q.assign([initial, rig.offset + rig.ratio * initial])
        newton.eval_fk(rig.model, rig.state.joint_q, rig.model.joint_qd, rig.state)
        rig.angles.fill_(initial)
        rig.step(0.0)
        rig.check()


def test_vbd_mimic_d6_coordinates(test, device):
    """Track D6 linear and angular coordinates independently, with differing q/qd offsets."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    # A preceding FREE joint ensures coordinate indices differ from velocity indices.
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
    joints = []
    for kinematic in (True, False):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=kinematic)
        cfg = newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.Z, target_ke=0.0, target_kd=0.0, limit_ke=0.0, limit_kd=0.0
        )
        joints.append(builder.add_joint_d6(-1, body, linear_axes=[cfg], angular_axes=[cfg]))
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0], (0.0, 0.5))
    initial = 4.0 * math.pi + 0.2
    builder.joint_q[-4:] = [2.0, initial, 1.0, 0.5 * initial]
    builder.color()
    model = builder.finalize(device=device)
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverVBD(model, iterations=12, rigid_compliant_alm=True)
    for step in range(200):
        linear, angular = 2.0 + 0.03 * step, initial + 0.1 * step
        wp.launch(_prescribe_d6_reference, dim=1, inputs=[linear, angular], outputs=[state.body_q], device=device)
        solver.step(state, next_state, None, None, 1.0 / 240.0)
        state, next_state = next_state, state
        pose = state.body_q.numpy()[2]
        test.assertAlmostEqual(float(pose[2]), 0.5 * linear, delta=2.0e-5)
        expected = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * angular))
        np.testing.assert_allclose(abs(np.dot(pose[3:], expected)), 1.0, atol=2.0e-6)


def test_vbd_mimic_masked_reset(test, device):
    """Reseed selected worlds after pose edits without disturbing other turn histories."""
    rig = _MimicRig(device, worlds=True)
    for _ in range(100):
        rig.step()
    for world in (0, 2):
        selected = np.zeros(3, dtype=bool)
        selected[world] = True
        mask = wp.array(selected, dtype=bool, device=device)
        rig.solver.reset(rig.state, world_mask=mask, flags=0)
        angle = -4.0 * math.pi - 0.2
        q = rig.state.joint_q.numpy()
        q[2 * world : 2 * world + 2] = [angle, rig.offset + rig.ratio * angle]
        rig.state.joint_q.assign(q)
        # Author after reset: the next step must use these poses and turn seeds.
        newton.eval_fk(rig.model, rig.state.joint_q, rig.model.joint_qd, rig.state, mask=mask)
        angles = rig.angles.numpy()
        angles[world] = angle
        rig.angles.assign(angles)
        rig.step()
        rig.check()

    rig.solver.reset(rig.state)
    rig.angles.zero_()
    rig.step()
    rig.check()


def test_vbd_mimic_disabled_history(test, device):
    """Keep counting turns while mimic enforcement is disabled and after notification."""
    rig = _MimicRig(device)
    for _ in range(100):
        rig.step()
    rig.model.joint_enabled.assign([False, True])
    rig.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    for _ in range(100):
        rig.step()
    rig.model.joint_enabled.fill_(True)
    rig.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    rig.step()
    rig.check()


def test_vbd_mimic_graph_replay(test, device):
    """Retain turn history across graph replays, including a captured episode reset."""
    rig = _MimicRig(device)
    rig.step(0.0)  # Load kernels before capture.
    rig.solver.reset(rig.state)
    graph = None
    reset_graph = None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            rig.step()
            rig.step()
        graph = capture.graph
        with wp.ScopedCapture(device=device) as capture:
            rig.solver.reset(rig.state)
            rig.angles.zero_()
        reset_graph = capture.graph
    for episode in range(2):
        if episode:
            if reset_graph is None:
                rig.solver.reset(rig.state)
                rig.angles.zero_()
            else:
                wp.capture_launch(reset_graph)
        for _ in range(100):
            if graph is None:
                rig.step()
                rig.step()
            else:
                wp.capture_launch(graph)
            rig.check()


class TestSolverVBDMimic(unittest.TestCase):
    pass


for test_function in (
    test_vbd_mimic_multiturn_translation,
    test_vbd_mimic_fractional_angular_ratio,
    test_vbd_mimic_multiturn_dynamics,
    test_vbd_mimic_initial_turns,
    test_vbd_mimic_d6_coordinates,
    test_vbd_mimic_masked_reset,
    test_vbd_mimic_disabled_history,
    test_vbd_mimic_graph_replay,
):
    add_function_test(TestSolverVBDMimic, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
