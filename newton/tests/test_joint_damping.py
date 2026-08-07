# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


class TestJointDamping(unittest.TestCase):
    pass


def _build_revolute_model(device, damping: float):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Z,
        target_ke=0.0,
        target_kd=0.0,
        damping=damping,
        limit_lower=-1.0e6,
        limit_upper=1.0e6,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=0.0,
        friction=0.0,
    )
    builder.add_articulation([joint])
    builder.joint_qd[0] = 1.0

    return builder.finalize(device=device)


def _build_ball_model(
    device,
    damping: float,
    initial_qd: tuple[float, float, float] = (1.0, 0.0, 0.0),
    use_public_helper: bool = False,
    parent_xform: wp.transform | None = None,
    child_xform: wp.transform | None = None,
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    if use_public_helper:
        joint = builder.add_joint_ball(
            parent=-1,
            child=body,
            parent_xform=parent_xform,
            child_xform=child_xform,
            damping=damping,
        )
    else:
        joint = builder.add_joint(
            newton.JointType.BALL,
            parent=-1,
            child=body,
            parent_xform=parent_xform,
            child_xform=child_xform,
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, damping=damping, armature=0.0, friction=0.0),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, damping=damping, armature=0.0, friction=0.0),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, damping=damping, armature=0.0, friction=0.0),
            ],
        )

    builder.add_articulation([joint])
    builder.joint_qd[0:3] = list(initial_qd)

    return builder.finalize(device=device)


def _simulate_joint_damping(device, solver_fn, damping: float, sync_joint_qd: bool) -> tuple[float, float]:
    model = _build_revolute_model(device, damping)
    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_qd = float(state_0.joint_qd.numpy()[0])
    for _ in range(8):
        solver.step(state_0, state_1, control, None, 0.01)
        state_0, state_1 = state_1, state_0

    if sync_joint_qd:
        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

    return initial_qd, float(state_0.joint_qd.numpy()[0])


def _simulate_ball_joint_damping(
    device,
    solver_fn,
    damping: float,
    initial_qd: tuple[float, float, float] = (1.0, 0.0, 0.0),
    use_public_helper: bool = False,
    parent_xform: wp.transform | None = None,
    child_xform: wp.transform | None = None,
) -> tuple[float, float]:
    model = _build_ball_model(
        device,
        damping,
        initial_qd=initial_qd,
        use_public_helper=use_public_helper,
        parent_xform=parent_xform,
        child_xform=child_xform,
    )
    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_speed = float(np.linalg.norm(state_0.joint_qd.numpy()[0:3]))
    for _ in range(8):
        solver.step(state_0, state_1, control, None, 0.01)
        state_0, state_1 = state_1, state_0

    newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
    return initial_speed, float(np.linalg.norm(state_0.joint_qd.numpy()[0:3]))


def test_revolute_joint_damping_decays_velocity(test: TestJointDamping, device, solver_fn, sync_joint_qd):
    undamped_initial, undamped_final = _simulate_joint_damping(
        device, solver_fn, damping=0.0, sync_joint_qd=sync_joint_qd
    )
    damped_initial, damped_final = _simulate_joint_damping(device, solver_fn, damping=3.0, sync_joint_qd=sync_joint_qd)

    np.testing.assert_allclose(undamped_final, undamped_initial, atol=1.0e-5, rtol=1.0e-5)
    test.assertLess(abs(damped_final), abs(damped_initial) * 0.85)


def test_semi_implicit_ball_joint_damping_decays_velocity(test: TestJointDamping, device):
    def solver_fn(model):
        return newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)

    cases = (
        ((1.0, 0.0, 0.0), False, None, None),
        ((0.0, 1.0, 0.0), False, None, None),
        ((0.0, 0.0, 1.0), False, None, None),
        ((0.5, -0.25, 1.0), False, None, None),
        (
            (0.5, -0.25, 1.0),
            True,
            wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3)),
            wp.transform_identity(),
        ),
    )

    for initial_qd, use_public_helper, parent_xform, child_xform in cases:
        with test.subTest(initial_qd=initial_qd, use_public_helper=use_public_helper):
            undamped_initial, undamped_final = _simulate_ball_joint_damping(
                device,
                solver_fn,
                damping=0.0,
                initial_qd=initial_qd,
                use_public_helper=use_public_helper,
                parent_xform=parent_xform,
                child_xform=child_xform,
            )
            damped_initial, damped_final = _simulate_ball_joint_damping(
                device,
                solver_fn,
                damping=3.0,
                initial_qd=initial_qd,
                use_public_helper=use_public_helper,
                parent_xform=parent_xform,
                child_xform=child_xform,
            )

            np.testing.assert_allclose(undamped_final, undamped_initial, atol=1.0e-5, rtol=1.0e-5)
            test.assertLess(damped_final, damped_initial * 0.85)


def test_featherstone_ball_joint_damping_decays_velocity(test: TestJointDamping, device):
    def solver_fn(model):
        return newton.solvers.SolverFeatherstone(model, angular_damping=0.0)

    undamped_initial, undamped_final = _simulate_ball_joint_damping(device, solver_fn, damping=0.0)
    damped_initial, damped_final = _simulate_ball_joint_damping(device, solver_fn, damping=3.0)

    np.testing.assert_allclose(undamped_final, undamped_initial, atol=1.0e-5, rtol=1.0e-5)
    test.assertLess(damped_final, damped_initial * 0.85)


def test_add_joint_ball_sets_passive_damping(test: TestJointDamping, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_ball(parent=-1, child=body, damping=2.5)
    builder.add_articulation([joint])
    model = builder.finalize(device=device)

    np.testing.assert_allclose(model.joint_damping.numpy()[0:3], [2.5, 2.5, 2.5])


def test_mjcf_ball_joint_damping_overrides_default(test: TestJointDamping, device):
    # Regression test for gh #3703: an MJCF <joint type="ball" damping="..."> must
    # forward its authored damping into joint_damping instead of silently falling
    # back to default_joint_cfg.damping. Before the fix, importing with a non-zero
    # default_joint_cfg.damping raised a ValueError at finalize() because the parsed
    # damping (recorded in the deprecated dof_passive_damping alias) disagreed with
    # the default that add_joint_ball wrote into joint_damping.
    xml = """
    <mujoco><worldbody><body name="base"><geom type="sphere" size="0.05" mass="1"/>
      <body name="link" pos="0.1 0 0"><joint name="ball" type="ball" damping="7"/>
      <geom type="sphere" size="0.04" mass="0.5"/></body></body></worldbody></mujoco>
    """
    for default_damping in (0.0, 99.0):
        builder = newton.ModelBuilder()
        builder.default_joint_cfg.damping = default_damping
        builder.add_mjcf(xml)
        model = builder.finalize(device=device)
        np.testing.assert_allclose(model.joint_damping.numpy()[0:3], [7.0, 7.0, 7.0])


def test_mjcf_ball_joint_damping_falls_back_to_default(test: TestJointDamping, device):
    # Companion to test_mjcf_ball_joint_damping_overrides_default: when the MJCF ball
    # joint does not author a damping, the builder default must still apply.
    xml = """
    <mujoco><worldbody><body name="base"><geom type="sphere" size="0.05" mass="1"/>
      <body name="link" pos="0.1 0 0"><joint name="ball" type="ball"/>
      <geom type="sphere" size="0.04" mass="0.5"/></body></body></worldbody></mujoco>
    """
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.damping = 3.5
    builder.add_mjcf(xml)
    model = builder.finalize(device=device)
    np.testing.assert_allclose(model.joint_damping.numpy()[0:3], [3.5, 3.5, 3.5])


devices = get_test_devices()
solvers = {
    "featherstone": (lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0), False),
    "semi_implicit": (lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0), True),
    "kamino": (newton.solvers.SolverKamino, False),
}

for device in devices:
    for solver_name, (solver_fn, sync_joint_qd) in solvers.items():
        add_function_test(
            TestJointDamping,
            f"test_revolute_joint_damping_decays_velocity_{solver_name}",
            test_revolute_joint_damping_decays_velocity,
            devices=[device],
            solver_fn=solver_fn,
            sync_joint_qd=sync_joint_qd,
        )

for device in devices:
    add_function_test(
        TestJointDamping,
        "test_semi_implicit_ball_joint_damping_decays_velocity",
        test_semi_implicit_ball_joint_damping_decays_velocity,
        devices=[device],
    )
    add_function_test(
        TestJointDamping,
        "test_featherstone_ball_joint_damping_decays_velocity",
        test_featherstone_ball_joint_damping_decays_velocity,
        devices=[device],
    )
    add_function_test(
        TestJointDamping,
        "test_add_joint_ball_sets_passive_damping",
        test_add_joint_ball_sets_passive_damping,
        devices=[device],
    )
    add_function_test(
        TestJointDamping,
        "test_mjcf_ball_joint_damping_overrides_default",
        test_mjcf_ball_joint_damping_overrides_default,
        devices=[device],
    )
    add_function_test(
        TestJointDamping,
        "test_mjcf_ball_joint_damping_falls_back_to_default",
        test_mjcf_ball_joint_damping_falls_back_to_default,
        devices=[device],
    )

for device in get_cuda_test_devices():
    add_function_test(
        TestJointDamping,
        "test_revolute_joint_damping_decays_velocity_mujoco_warp",
        test_revolute_joint_damping_decays_velocity,
        devices=[device],
        solver_fn=lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
        sync_joint_qd=False,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
