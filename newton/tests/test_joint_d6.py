# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.joint_mimic import eval_joint_mimic_coordinate, eval_joint_mimic_velocity
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Non-XYZ axis orders, as in the nv_humanoid.xml hips.
AXES_3 = (newton.Axis.X, newton.Axis.Z, newton.Axis.Y)
AXES_2 = (newton.Axis.X, newton.Axis.Z)
Q0 = np.array([0.5, -0.4, 0.7], dtype=np.float32)


class TestJointD6(unittest.TestCase):
    pass


def _build(device, axes, **dof_kwargs):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.1, 0.2, 0.15])))
    joint = builder.add_joint_d6(
        -1,
        body,
        angular_axes=[newton.ModelBuilder.JointDofConfig(axis=axis, armature=0.0, **dof_kwargs) for axis in axes],
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    state = model.state()
    state.joint_q.assign(Q0[: len(axes)])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _step(model, solver, state, control, steps=10, dt=1.0e-3):
    state_1 = model.state()
    for _ in range(steps):
        state.clear_forces()
        solver.step(state, state_1, control, None, dt)
        state, state_1 = state_1, state
    newton.eval_ik(model, state, state.joint_q, state.joint_qd)
    return state


def test_d6_angular_target_is_equilibrium(test, device, solver_fn):
    """A body posed by eval_fk at the drive targets must stay at rest."""
    for axes in (AXES_3, AXES_2):
        with test.subTest(axis_count=len(axes)):
            model, state = _build(device, axes, target_ke=1000.0, target_kd=10.0)
            control = model.control()
            control.joint_target_q.assign(Q0[: len(axes)])
            state = _step(model, solver_fn(model), state, control)
            np.testing.assert_allclose(state.joint_q.numpy(), Q0[: len(axes)], atol=1.0e-4)


def test_d6_angular_joint_f_matches_featherstone(test, device, solver_fn):
    """A generalized force on one angular coordinate must not drive the others."""
    joint_f = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    joint_qd = []
    for make_solver in (newton.solvers.SolverFeatherstone, solver_fn):
        model, state = _build(device, AXES_3)
        control = model.control()
        control.joint_f.assign(joint_f)
        joint_qd.append(_step(model, make_solver(model), state, control).joint_qd.numpy())
    np.testing.assert_allclose(joint_qd[1], joint_qd[0], atol=2.0e-3)


@wp.kernel
def _mimic_coordinate_velocities(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    coordinates: wp.array[float],
    velocities: wp.array[float],
):
    k = wp.tid()
    q, grad_p, grad_c = eval_joint_mimic_coordinate(
        0,
        k,
        body_q,
        body_com,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_qd_start,
        joint_dof_dim,
        joint_axis,
    )
    coordinates[k] = q
    velocities[k] = eval_joint_mimic_velocity(joint_parent[0], joint_child[0], grad_p, grad_c, body_qd)


def test_d6_mimic_coordinate_gradients(test, device):
    """Mimic coordinate gradients must map body velocities to the joint velocities used by eval_fk."""
    for axes in (AXES_3, AXES_2):
        with test.subTest(axis_count=len(axes)):
            model, state = _build(device, axes)
            qd = np.array([0.9, -0.6, 0.3], dtype=np.float32)[: len(axes)]
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            coordinates = wp.zeros(len(axes), dtype=float, device=device)
            velocities = wp.zeros(len(axes), dtype=float, device=device)
            wp.launch(
                _mimic_coordinate_velocities,
                dim=len(axes),
                inputs=[
                    state.body_q,
                    state.body_qd,
                    model.body_com,
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_axis,
                ],
                outputs=[coordinates, velocities],
                device=device,
            )
            np.testing.assert_allclose(coordinates.numpy(), Q0[: len(axes)], atol=1.0e-5)
            np.testing.assert_allclose(velocities.numpy(), qd, atol=1.0e-5)


solvers = {
    "xpbd": newton.solvers.SolverXPBD,
    "semi_implicit": newton.solvers.SolverSemiImplicit,
}

devices = get_test_devices()
for solver_name, solver_fn in solvers.items():
    add_function_test(
        TestJointD6,
        f"test_d6_angular_target_is_equilibrium_{solver_name}",
        test_d6_angular_target_is_equilibrium,
        devices=devices,
        solver_fn=solver_fn,
    )
    add_function_test(
        TestJointD6,
        f"test_d6_angular_joint_f_matches_featherstone_{solver_name}",
        test_d6_angular_joint_f_matches_featherstone,
        devices=devices,
        solver_fn=solver_fn,
    )
add_function_test(
    TestJointD6, "test_d6_mimic_coordinate_gradients", test_d6_mimic_coordinate_gradients, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
