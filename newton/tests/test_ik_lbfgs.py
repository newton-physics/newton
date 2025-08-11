# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
import newton.sim.ik as ik
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_two_link_planar(device) -> newton.Model:
    """Returns a singleton model with one 2-DOF planar arm."""
    builder = newton.ModelBuilder()

    link1 = builder.add_body(
        xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    builder.add_joint_revolute(
        parent=-1,
        child=link1,
        parent_xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )

    link2 = builder.add_body(
        xform=wp.transform([1.5, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )

    model = builder.finalize(device=device, requires_grad=True)
    return model


def _fk_end_effector_positions(
    model: newton.Model, body_q_2d: wp.array, n_problems: int, ee_link_index: int, ee_offset: wp.vec3
) -> np.ndarray:
    """Returns an (N,3) array with end-effector world positions for every problem."""
    positions = np.zeros((n_problems, 3), dtype=np.float32)
    body_q_np = body_q_2d.numpy()  # shape: [n_problems, model.body_count]

    for prob in range(n_problems):
        body_tf = body_q_np[prob, ee_link_index]
        pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
        rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
        ee_world = wp.transform_point(wp.transform(pos, rot), ee_offset)
        positions[prob] = [ee_world[0], ee_world[1], ee_world[2]]
    return positions


def _convergence_test_lbfgs_planar(test, device, mode: ik.JacobianMode):
    """Test L-BFGS convergence on planar 2-link robot."""
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_two_link_planar(device)

        # Create 2D joint_q array [n_problems, joint_coord_count]
        requires_grad = mode in [ik.JacobianMode.AUTODIFF, ik.JacobianMode.MIXED]
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)

        # Create 2D joint_qd array [n_problems, joint_dof_count]
        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)

        # Create 2D body arrays for output
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        # Reachable XY targets
        targets = wp.array([[1.5, 1.0, 0.0], [1.2, 0.8, 0.0], [1.8, 0.5, 0.0]], dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.PositionObjective(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
            n_problems=n_problems,
            total_residuals=3,
            residual_offset=0,
        )

        # Create L-BFGS solver
        lbfgs_solver = ik.LBFGSSolver(
            model,
            joint_q_2d,
            [pos_obj],
            jacobian_mode=mode,
        )

        # Run initial FK
        ik._eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        # Solve with L-BFGS
        lbfgs_solver.solve(iterations=70)

        # Run final FK
        ik._eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        # Check convergence
        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err1, err0, f"L-BFGS mode {mode} problem {prob} did not improve")
            test.assertLess(err1, 3e-3, f"L-BFGS mode {mode} problem {prob} final error too high ({err1:.4f})")


def _comparison_test_lm_vs_lbfgs(test, device, mode: ik.JacobianMode):
    """Compare L-BFGS vs LM solver performance."""
    with wp.ScopedDevice(device):
        n_problems = 2
        model = _build_two_link_planar(device)

        requires_grad = mode in [ik.JacobianMode.AUTODIFF, ik.JacobianMode.MIXED]

        # Create identical initial conditions for both solvers
        joint_q_lm = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)
        joint_q_lbfgs = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)

        # Set challenging initial configuration
        initial_q = np.array([[0.5, -0.8], [0.3, 1.2]], dtype=np.float32)
        joint_q_lm.assign(initial_q)
        joint_q_lbfgs.assign(initial_q)

        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        # Challenging targets
        targets = wp.array([[1.4, 1.2, 0.0], [1.0, 1.5, 0.0]], dtype=wp.vec3)
        ee_link, ee_off = 1, wp.vec3(0.5, 0.0, 0.0)

        # Create objectives
        pos_obj_lm = ik.PositionObjective(ee_link, ee_off, targets, n_problems, 3, 0)
        pos_obj_lbfgs = ik.PositionObjective(ee_link, ee_off, targets, n_problems, 3, 0)

        # Create solvers
        lm_solver = ik.IKSolver(model, joint_q_lm, [pos_obj_lm], lambda_initial=1e-3, jacobian_mode=mode)
        lbfgs_solver = ik.LBFGSSolver(model, joint_q_lbfgs, [pos_obj_lbfgs], jacobian_mode=mode, history_len=8)

        # Get initial errors
        ik._eval_fk_batched(model, joint_q_lm, joint_qd_2d, body_q_2d, body_qd_2d)
        initial_lm = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        ik._eval_fk_batched(model, joint_q_lbfgs, joint_qd_2d, body_q_2d, body_qd_2d)
        initial_lbfgs = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        # Solve with both methods
        lm_solver.solve(iterations=25)
        lbfgs_solver.solve(iterations=70)

        # Get final errors
        ik._eval_fk_batched(model, joint_q_lm, joint_qd_2d, body_q_2d, body_qd_2d)
        final_lm = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        ik._eval_fk_batched(model, joint_q_lbfgs, joint_qd_2d, body_q_2d, body_qd_2d)
        final_lbfgs = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        # Both solvers should converge
        for prob in range(n_problems):
            target = targets.numpy()[prob]

            err_lm_initial = np.linalg.norm(initial_lm[prob] - target)
            err_lm_final = np.linalg.norm(final_lm[prob] - target)

            err_lbfgs_initial = np.linalg.norm(initial_lbfgs[prob] - target)
            err_lbfgs_final = np.linalg.norm(final_lbfgs[prob] - target)

            # Both should improve
            test.assertLess(err_lm_final, err_lm_initial, f"LM problem {prob} did not improve")
            test.assertLess(err_lbfgs_final, err_lbfgs_initial, f"L-BFGS problem {prob} did not improve")

            # Both should achieve good accuracy
            test.assertLess(err_lm_final, 1e-3, f"LM problem {prob} final error too high ({err_lm_final:.4f})")
            test.assertLess(
                err_lbfgs_final, 1e-3, f"L-BFGS problem {prob} final error too high ({err_lbfgs_final:.4f})"
            )


# Test functions
def test_lbfgs_convergence_autodiff(test, device):
    _convergence_test_lbfgs_planar(test, device, ik.JacobianMode.AUTODIFF)


def test_lbfgs_convergence_analytic(test, device):
    _convergence_test_lbfgs_planar(test, device, ik.JacobianMode.ANALYTIC)


def test_lbfgs_convergence_mixed(test, device):
    _convergence_test_lbfgs_planar(test, device, ik.JacobianMode.MIXED)


def test_lm_vs_lbfgs_comparison_autodiff(test, device):
    _comparison_test_lm_vs_lbfgs(test, device, ik.JacobianMode.AUTODIFF)


def test_lm_vs_lbfgs_comparison_analytic(test, device):
    _comparison_test_lm_vs_lbfgs(test, device, ik.JacobianMode.ANALYTIC)


def test_lm_vs_lbfgs_comparison_mixed(test, device):
    _comparison_test_lm_vs_lbfgs(test, device, ik.JacobianMode.MIXED)


# Test registration
devices = get_test_devices()


class TestLBFGSIK(unittest.TestCase):
    pass


# Register L-BFGS convergence tests
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_autodiff", test_lbfgs_convergence_autodiff, devices)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_analytic", test_lbfgs_convergence_analytic, devices)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_mixed", test_lbfgs_convergence_mixed, devices)

# Register comparison tests
add_function_test(TestLBFGSIK, "test_lm_vs_lbfgs_comparison_autodiff", test_lm_vs_lbfgs_comparison_autodiff, devices)
add_function_test(TestLBFGSIK, "test_lm_vs_lbfgs_comparison_analytic", test_lm_vs_lbfgs_comparison_analytic, devices)
add_function_test(TestLBFGSIK, "test_lm_vs_lbfgs_comparison_mixed", test_lm_vs_lbfgs_comparison_mixed, devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
