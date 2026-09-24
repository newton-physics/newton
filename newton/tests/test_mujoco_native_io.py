# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise native MuJoCo CUDA transfers without changing CPU integration."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _model(device):
    """Build a free body whose force response has an analytic reference."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0))
        return builder.finalize()


def _run_native_io(test, device):
    """Preserve changed forces, clearing, state bindings, and model updates."""
    model = _model(device)
    reference = SolverMuJoCo(model, use_mujoco_cpu=True, update_data_interval=0)
    captured = SolverMuJoCo(model, use_mujoco_cpu=True, update_data_interval=0)
    r0, r1, c0, c1 = [model.state() for _ in range(4)]
    control = model.control()
    if wp.get_device(device).is_cuda:
        captured.capture_native_io(c0, c1, control)
        captured.capture_native_io(c1, c0, control)
    else:
        with test.assertRaisesRegex(ValueError, "CUDA model"):
            captured.capture_native_io(c0, c1, control)
    for i in range(12):
        for state in (r0, c0):
            state.clear_forces()
            if i == 0:
                state.body_f.assign(np.array([[2, 0, 0, 0, 0, 0]], np.float32))
        control.joint_f.zero_()
        if i == 4:
            control.joint_f.assign(np.array([0, 3, 0, 0, 0, 0], np.float32))
        reference.step(r0, r1, control, None, 0.01)
        captured.step(c0, c1, control, None, 0.01)
        r0, r1, c0, c1 = r1, r0, c1, c0
        np.testing.assert_array_equal(c0.body_q.numpy(), r0.body_q.numpy())
        np.testing.assert_array_equal(c0.body_qd.numpy(), r0.body_qd.numpy())
    np.testing.assert_allclose(c0.body_qd.numpy()[0, :3], [0.02, 0.03, 0], atol=1e-7)
    for field in ("xfrc_applied", "qfrc_applied"):
        np.testing.assert_array_equal(getattr(captured.mj_data, field), 0)
    if wp.get_device(device).is_cuda:
        # A new binding must use freshly packed arguments, not an old graph.
        with patch.object(captured, "_apply_mjc_control_gpu", wraps=captured._apply_mjc_control_gpu) as apply:
            new_control = model.control()
            captured.step(c0, c1, new_control, None, 0.01)
            test.assertEqual(apply.call_count, 1)
            captured.step(model.state(), model.state(), control, None, 0.01)
            test.assertEqual(apply.call_count, 2)
        captured.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        test.assertEqual(len(captured._native_io_graphs), 0)
        with patch.object(captured, "_apply_mjc_control_gpu", wraps=captured._apply_mjc_control_gpu) as apply:
            captured.step(c0, c1, control, None, 0.01)
            test.assertEqual(apply.call_count, 1)


class TestMuJoCoNativeIO(unittest.TestCase):
    def test_native_model_update_preserves_integration_state(self):
        """Preserve live state when refreshing gains with or without kinematic bodies."""
        with wp.ScopedDevice("cpu"):
            for is_kinematic in (False, True):
                with self.subTest(is_kinematic=is_kinematic):
                    builder = newton.ModelBuilder(gravity=(0, 0, 0))
                    cfg = newton.ModelBuilder.ShapeConfig(density=0)
                    free_body = builder.add_body(
                        mass=1.0,
                        inertia=wp.mat33(np.eye(3)),
                        lock_inertia=True,
                        is_kinematic=is_kinematic,
                    )
                    builder.add_shape_sphere(free_body, radius=0.1, cfg=cfg)
                    driven_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                    builder.add_shape_sphere(driven_body, radius=0.1, cfg=cfg)
                    joint = builder.add_joint_revolute(
                        -1,
                        driven_body,
                        axis=(0, 0, 1),
                        target_ke=45.0,
                        target_kd=4.5,
                        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                    )
                    builder.add_articulation([joint])
                    model = builder.finalize()
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True, update_data_interval=0, disable_contacts=True)
                    native, data, mujoco = solver.mj_model, solver.mj_data, solver._mujoco
                    free_joint = np.flatnonzero(native.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
                    free_qpos = native.jnt_qposadr[free_joint]
                    data.qpos[free_qpos : free_qpos + 3] = [0.3, -0.2, 0.4]
                    data.qpos[free_qpos + 3 : free_qpos + 7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
                    data.qvel[:] = np.linspace(0.1, 0.2, native.nv)
                    data.time = 1.25
                    data.ctrl[:] = 0.4
                    data.qfrc_applied[:] = np.linspace(0.2, 0.3, native.nv)
                    data.xfrc_applied[:] = 0.1
                    mujoco.mj_forward(native, data)
                    data.qacc_warmstart[:] = np.linspace(0.3, 0.4, native.nv)
                    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
                    before = np.empty(mujoco.mj_stateSize(native, state_spec))
                    mujoco.mj_getState(native, data, before, state_spec)
                    derived = {name: getattr(data, name).copy() for name in ("qacc", "xpos", "xquat", "actuator_force")}

                    gains = model.joint_target_ke.numpy().copy()
                    gains[gains > 0] = 150.0
                    model.joint_target_ke.assign(gains)
                    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

                    after = np.empty_like(before)
                    mujoco.mj_getState(native, data, after, state_spec)
                    np.testing.assert_array_equal(after, before)
                    for name, expected in derived.items():
                        np.testing.assert_array_equal(getattr(data, name), expected, err_msg=name)
                    self.assertEqual(native.actuator_gainprm[0, 0], 150.0)
                    self.assertEqual(native.actuator_biasprm[0, 1], -150.0)

    def test_native_joint_target_properties_update(self):
        """Apply changed target gains to native forces while retaining effort limits."""
        with wp.ScopedDevice("cpu"):
            for mode in (newton.JointTargetMode.POSITION, newton.JointTargetMode.POSITION_VELOCITY):
                with self.subTest(mode=mode):
                    builder = newton.ModelBuilder(gravity=(0, 0, 0))
                    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                    builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0))
                    joint = builder.add_joint_revolute(
                        -1,
                        body,
                        axis=(0, 0, 1),
                        target_ke=45.0,
                        target_kd=4.5,
                        actuator_mode=mode,
                        effort_limit=16.5,
                    )
                    builder.add_articulation([joint])
                    model = builder.finalize()
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                    native, data = solver.mj_model, solver.mj_data
                    original_limits = native.jnt_actfrcrange.copy()
                    self.assertTrue(native.jnt_actfrclimited[0])
                    np.testing.assert_array_equal(original_limits[0], [-16.5, 16.5])

                    for ke, kd in ((150.0, 4.5), (150.0, 7.0)):
                        model.joint_target_ke.assign(np.array([ke], dtype=np.float32))
                        model.joint_target_kd.assign(np.array([kd], dtype=np.float32))
                        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                        np.testing.assert_allclose(
                            native.actuator_gainprm, solver.mjw_model.actuator_gainprm.numpy()[0]
                        )
                        np.testing.assert_allclose(
                            native.actuator_biasprm, solver.mjw_model.actuator_biasprm.numpy()[0]
                        )
                        self.assertEqual(native.actuator_gainprm[0, 0], ke)
                        self.assertEqual(native.actuator_biasprm[0, 1], -ke)
                        np.testing.assert_array_equal(native.jnt_actfrcrange, original_limits)
                        self.assertTrue(native.jnt_actfrclimited[0])

                        data.qpos[0], data.qvel[0] = 0.0, 0.2
                        data.ctrl[:] = 0.0
                        data.ctrl[0] = 0.1
                        solver._mujoco.mj_forward(native, data)
                        self.assertAlmostEqual(data.qfrc_actuator[0], ke * 0.1 - kd * 0.2, places=6)
                        data.ctrl[0] = 10.0
                        solver._mujoco.mj_forward(native, data)
                        self.assertAlmostEqual(data.qfrc_actuator[0], 16.5, places=6)


def _run_native_actuator_properties_update(test, device):
    """Apply native actuator edits and limits independently of notification order."""
    xml = """<mujoco>
        <option gravity="0 0 0"/>
        <worldbody><body>
            <joint name="hinge" type="hinge"/>
            <geom type="sphere" size="0.1" mass="1"/>
        </body></worldbody>
        <actuator>
            <general joint="hinge" gainprm="50" biastype="affine" biasprm="1 -2 -3"
                dyntype="filter" dynprm="0.1" gear="1" cranklength="1"
                ctrllimited="true" ctrlrange="-1 1" forcelimited="true" forcerange="-100 100"
                actlimited="true" actrange="-1 1"/>
        </actuator>
    </mujoco>"""
    actuator_flags = newton.ModelFlags.ACTUATOR_PROPERTIES
    joint_flags = newton.ModelFlags.JOINT_DOF_PROPERTIES
    updates = {
        "actuator_gainprm": [150.0] + [0.0] * 9,
        "actuator_biasprm": [2.0, -4.0, -6.0] + [0.0] * 7,
        "actuator_dynprm": [0.25] + [0.0] * 9,
        "actuator_ctrlrange": [-0.5, 0.5],
        "actuator_forcerange": [-20.0, 20.0],
        "actuator_actrange": [-0.75, 0.75],
        "actuator_gear": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "actuator_cranklength": 1.5,
    }
    with wp.ScopedDevice(device):
        for notifications in (
            (actuator_flags,),
            (joint_flags, actuator_flags),
            (actuator_flags, joint_flags),
            (actuator_flags | joint_flags,),
        ):
            with test.subTest(notifications=notifications):
                builder = newton.ModelBuilder()
                builder.add_mjcf(xml, ctrl_direct=True)
                model = builder.finalize()
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                native, data = solver.mj_model, solver.mj_data
                for field, value in updates.items():
                    array = getattr(model.mujoco, field)
                    values = array.numpy()
                    values[0] = value
                    array.assign(values)

                for flags in notifications:
                    solver.notify_model_changed(flags)

                for field, expected in updates.items():
                    with test.subTest(field=field):
                        np.testing.assert_allclose(getattr(native, field)[0], expected)

                # The gear scales both actuator length/velocity and joint torque.
                data.qpos[0], data.qvel[0] = 0.1, 0.2
                for activation, control, force, act_dot in (
                    (0.1, 0.2, 13.8, 0.4),
                    (0.3, 1.0, 20.0, 0.8),
                    (-0.3, -1.0, -20.0, -0.8),
                ):
                    data.act[0], data.ctrl[0] = activation, control
                    solver._mujoco.mj_forward(native, data)
                    test.assertAlmostEqual(data.actuator_force[0], force, places=6)
                    test.assertAlmostEqual(data.qfrc_actuator[0], 2.0 * force, places=6)
                    test.assertAlmostEqual(data.act_dot[0], act_dot, places=6)

                data.act[0], data.ctrl[0] = 0.8, 0.0
                solver._mujoco.mj_step(native, data)
                test.assertAlmostEqual(data.act[0], 0.75, places=6)


def _run_no_force_input(test, device):
    """Wait for pinned coordinate reads even when no force arrays are supplied."""
    model = _model(device)
    model.joint_qd.assign(np.array([0.5, 0, 0, 0, 0, 0], np.float32))
    solver = SolverMuJoCo(model, use_mujoco_cpu=True, update_data_interval=0)
    a, b = model.state(), model.state()
    a.body_f = b.body_f = None
    if wp.get_device(device).is_cuda:
        solver.capture_native_io(a, b, None)
        solver.capture_native_io(b, a, None)
    for _ in range(100):
        solver.step(a, b, None, None, 0.002)
        a, b = b, a
    np.testing.assert_allclose(a.body_q.numpy()[0, :3], [0.1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(a.body_qd.numpy()[0, :3], [0.5, 0, 0], atol=1e-7)


add_function_test(TestMuJoCoNativeIO, "test_native_io", _run_native_io, devices=get_test_devices())
add_function_test(
    TestMuJoCoNativeIO,
    "test_native_actuator_properties_update",
    _run_native_actuator_properties_update,
    devices=get_test_devices(),
)
add_function_test(TestMuJoCoNativeIO, "test_no_force_input", _run_no_force_input, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
