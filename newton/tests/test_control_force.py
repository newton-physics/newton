# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestControlForce(unittest.TestCase):
    pass


def _build_rotated_parent_descendant_free_model(device, child_com=None):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    if child_com is None:
        child_com = wp.vec3()

    base = builder.add_link(is_kinematic=True, mass=1.0)
    child = builder.add_link(mass=1.0)
    builder.add_shape_sphere(base, radius=0.1)
    builder.add_shape_sphere(child, radius=0.1)
    builder.body_com[child] = child_com

    j0 = builder.add_joint_fixed(
        parent=-1,
        child=base,
        parent_xform=wp.transform(
            wp.vec3(),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        ),
        child_xform=wp.transform_identity(),
    )
    j1 = builder.add_joint_free(parent=base, child=child)
    builder.add_articulation([j0, j1])

    return builder.finalize(device=device), base, child


def _rotate_into_parent_frame(q_parent_xyzw: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    q_parent = wp.quat(*q_parent_xyzw.astype(np.float32))
    v_parent = wp.quat_rotate_inv(q_parent, wp.vec3(*v_world.astype(np.float32)))
    return np.array(v_parent, dtype=np.float32)


def test_floating_body(test: TestControlForce, device, solver_fn, test_angular=True):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)

    # easy case: identity transform, zero center of mass
    b = builder.add_body()
    builder.add_shape_box(b)
    builder.joint_q = [1.0, 2.0, 3.0, *wp.quat_rpy(-1.3, 0.8, 2.4)]

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    if test_angular:
        control.joint_f.assign(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0], dtype=np.float32))
        test_index = 5
    else:
        control.joint_f.assign(np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        test_index = 1

    sim_dt = 1.0 / 10.0

    for _ in range(4):
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

    body_qd = state_0.body_qd.numpy()[0]
    test.assertGreater(body_qd[test_index], 0.04)
    test.assertLess(body_qd[test_index], 0.4)
    for i in range(6):
        if i == test_index:
            continue
        test.assertAlmostEqual(body_qd[i], 0.0, delta=1e-6)
    # TODO test joint_qd for MJC, Featherstone solvers


def test_3d_articulation(test: TestControlForce, device, solver_fn):
    # test mechanism with 3 orthogonally aligned prismatic joints
    # which allows to test all 3 dimensions of the control force independently
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.density = 100.0

    b = builder.add_link()
    builder.add_shape_sphere(b)
    j = builder.add_joint_d6(
        -1,
        b,
        linear_axes=[
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, armature=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, armature=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, armature=0.0),
        ],
    )
    builder.add_articulation([j])

    model = builder.finalize(device=device)

    test.assertEqual(model.joint_dof_count, 3)

    for control_dim in range(3):
        solver = solver_fn(model)

        state_0, state_1 = model.state(), model.state()

        control = model.control()
        control_input = np.zeros(model.joint_dof_count, dtype=np.float32)
        control_input[control_dim] = 100.0
        control.joint_f.assign(control_input)

        sim_dt = 1.0 / 10.0

        for _ in range(4):
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

        if not isinstance(solver, newton.solvers.SolverMuJoCo | newton.solvers.SolverFeatherstone):
            # need to compute joint_qd from body_qd
            newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

        qd = state_0.joint_qd.numpy()
        test.assertGreater(qd[control_dim], 0.009)
        test.assertLess(qd[control_dim], 0.4)
        for i in range(model.joint_dof_count):
            if i == control_dim:
                continue
            test.assertAlmostEqual(qd[i], 0.0, delta=1e-6)


def test_child_xform_moment_arm(test: TestControlForce, device, solver_fn):
    """Regression test for issue #1261: apply_joint_forces must include child joint transform.

    When a joint has a non-identity child_xform, a linear control force applied at the
    joint anchor should produce torque on the child body due to the moment arm between
    the joint anchor and the body COM.
    """
    offset_y = 2.0
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.density = 100.0

    b = builder.add_link()
    builder.add_shape_sphere(b)
    j = builder.add_joint_d6(
        -1,
        b,
        child_xform=((0.0, offset_y, 0.0), (0.0, 0.0, 0.0, 1.0)),
        linear_axes=[
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, armature=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, armature=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, armature=0.0),
        ],
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, armature=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, armature=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, armature=0.0),
        ],
    )
    builder.add_articulation([j])

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()

    control = model.control()
    # Apply force along X: with child_xform offset in Y, this should produce torque around Z
    control_input = np.zeros(model.joint_dof_count, dtype=np.float32)
    control_input[0] = 100.0  # force along X
    control.joint_f.assign(control_input)

    sim_dt = 1.0 / 10.0

    for _ in range(4):
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

    # body_qd layout: [vel_x, vel_y, vel_z, omega_x, omega_y, omega_z]
    body_qd = state_0.body_qd.numpy()[0]

    # The force along X should produce linear velocity along X
    test.assertGreater(body_qd[0], 0.001)

    # cross((0, offset_y, 0), (F, 0, 0)) = (0, 0, -F*offset_y)
    # So we expect negative angular velocity around Z
    test.assertLess(body_qd[5], -0.001, "Expected angular velocity around Z due to child_xform offset")


def test_featherstone_free_descendant_joint_f_world_wrench_preserves_public_frames(test: TestControlForce, device):
    """A descendant FREE world force should move in world coordinates while returned joint_qd stays parent-frame."""
    model, base, child = _build_rotated_parent_descendant_free_model(device, child_com=wp.vec3(0.2, -0.1, 0.05))
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    free_dof_start = model.joint_qd_start.numpy()[1]
    wrench = np.zeros(model.joint_dof_count, dtype=np.float32)
    wrench[free_dof_start : free_dof_start + 6] = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    control.joint_f.assign(wrench)

    solver.step(state_0, state_1, control, None, 0.01)

    body_qd = state_1.body_qd.numpy()[child]
    base_q = state_1.body_q.numpy()[base]
    expected_joint_qd = np.zeros(6, dtype=np.float32)
    expected_joint_qd[:3] = _rotate_into_parent_frame(base_q[3:7], body_qd[:3])
    expected_joint_qd[3:6] = _rotate_into_parent_frame(base_q[3:7], body_qd[3:6])

    test.assertGreater(body_qd[0], 1.0e-2)
    test.assertLess(np.linalg.norm(body_qd[1:3]), 1.0e-6)
    test.assertLess(np.linalg.norm(body_qd[3:6]), 1.0e-6)
    np.testing.assert_allclose(
        state_1.joint_qd.numpy()[free_dof_start : free_dof_start + 6],
        expected_joint_qd,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_featherstone_free_descendant_joint_f_world_torque_preserves_public_frames(test: TestControlForce, device):
    """A descendant FREE world torque should rotate in world coordinates while returned angular joint_qd stays parent-frame."""
    model, base, child = _build_rotated_parent_descendant_free_model(device, child_com=wp.vec3(0.2, -0.1, 0.05))
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    free_dof_start = model.joint_qd_start.numpy()[1]
    wrench = np.zeros(model.joint_dof_count, dtype=np.float32)
    wrench[free_dof_start : free_dof_start + 6] = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float32)
    control.joint_f.assign(wrench)

    solver.step(state_0, state_1, control, None, 0.01)

    body_qd = state_1.body_qd.numpy()[child]
    base_q = state_1.body_q.numpy()[base]
    expected_joint_qd = np.zeros(6, dtype=np.float32)
    expected_joint_qd[:3] = _rotate_into_parent_frame(base_q[3:7], body_qd[:3])
    expected_joint_qd[3:6] = _rotate_into_parent_frame(base_q[3:7], body_qd[3:6])

    test.assertLess(np.linalg.norm(body_qd[:3]), 3.0e-3)
    test.assertGreater(body_qd[3], 0.0)
    test.assertLess(np.linalg.norm(body_qd[4:6]), 1.0e-5)
    np.testing.assert_allclose(
        state_1.joint_qd.numpy()[free_dof_start : free_dof_start + 6],
        expected_joint_qd,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_featherstone_free_descendant_joint_f_does_not_require_prefk(test: TestControlForce, device):
    """The descendant FREE world-wrench path should not require a caller-side pre-step eval_fk()."""
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    base = builder.add_link(is_kinematic=True, mass=1.0)
    child = builder.add_link(mass=1.0)
    builder.add_shape_sphere(base, radius=0.1)
    builder.add_shape_box(
        child,
        xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity()),
        hx=0.1,
        hy=0.1,
        hz=0.1,
    )

    j0 = builder.add_joint_fixed(
        parent=-1,
        child=base,
        parent_xform=wp.transform(
            wp.vec3(1.0, 0.0, 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        ),
        child_xform=wp.transform_identity(),
    )
    j1 = builder.add_joint_free(parent=base, child=child)
    builder.add_articulation([j0, j1])

    model = builder.finalize(device=device)
    control = model.control()
    free_dof_start = model.joint_qd_start.numpy()[1]
    wrench = np.zeros(model.joint_dof_count, dtype=np.float32)
    wrench[free_dof_start : free_dof_start + 6] = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    control.joint_f.assign(wrench)

    solver_stale = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    stale_0, stale_1 = model.state(), model.state()
    solver_stale.step(stale_0, stale_1, control, None, 0.01)

    solver_fresh = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    fresh_0, fresh_1 = model.state(), model.state()
    newton.eval_fk(model, fresh_0.joint_q, fresh_0.joint_qd, fresh_0)
    solver_fresh.step(fresh_0, fresh_1, control, None, 0.01)

    np.testing.assert_allclose(stale_1.body_qd.numpy()[child], fresh_1.body_qd.numpy()[child], atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(stale_1.joint_qd.numpy(), fresh_1.joint_qd.numpy(), atol=1.0e-6, rtol=1.0e-6)
    test.assertGreater(stale_1.body_qd.numpy()[child][0], 1.0e-3)
    np.testing.assert_allclose(stale_1.body_qd.numpy()[child][3:6], 0.0, atol=1.0e-6, rtol=1.0e-6)


devices = get_test_devices()
solvers = {
    "featherstone": lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(
        model, use_mujoco_cpu=True, update_data_interval=0, disable_contacts=True
    ),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(
        model, use_mujoco_cpu=False, update_data_interval=0, disable_contacts=True
    ),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0),
    "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        # add_function_test(TestControlForce, f"test_floating_body_linear_{solver_name}", test_floating_body, devices=[device], solver_fn=solver_fn, test_angular=False)
        add_function_test(
            TestControlForce,
            f"test_floating_body_angular_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
        )
        add_function_test(
            TestControlForce,
            f"test_3d_articulation_{solver_name}",
            test_3d_articulation,
            devices=[device],
            solver_fn=solver_fn,
        )

# Only test solvers that use apply_joint_forces with child transform
child_xform_solvers = {
    "xpbd": solvers["xpbd"],
    "semi_implicit": solvers["semi_implicit"],
}
for device in devices:
    for solver_name, solver_fn in child_xform_solvers.items():
        add_function_test(
            TestControlForce,
            f"test_child_xform_moment_arm_{solver_name}",
            test_child_xform_moment_arm,
            devices=[device],
            solver_fn=solver_fn,
        )
    add_function_test(
        TestControlForce,
        "test_featherstone_free_descendant_joint_f_world_wrench_preserves_public_frames",
        test_featherstone_free_descendant_joint_f_world_wrench_preserves_public_frames,
        devices=[device],
    )
    add_function_test(
        TestControlForce,
        "test_featherstone_free_descendant_joint_f_world_torque_preserves_public_frames",
        test_featherstone_free_descendant_joint_f_world_torque_preserves_public_frames,
        devices=[device],
    )
    add_function_test(
        TestControlForce,
        "test_featherstone_free_descendant_joint_f_does_not_require_prefk",
        test_featherstone_free_descendant_joint_f_does_not_require_prefk,
        devices=[device],
    )

if __name__ == "__main__":
    unittest.main(verbosity=2)
