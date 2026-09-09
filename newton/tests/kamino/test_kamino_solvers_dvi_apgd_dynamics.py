# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end dynamics regressions for Kamino's associated-contact APGD path."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.geometry import inertia
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino.utils.extract import extract_problem_vector
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestDVIAPGDDynamics(unittest.TestCase):
    """Validate APGD in redundant, coupled, and compliant dynamics scenes."""

    pass


def _apgd_config(
    *,
    coupling_iterations: int = 12,
    post_stabilization_bilateral: bool = True,
    tolerance: float = 2.0e-4,
    apgd_max_iterations: int = 100,
) -> kamino_config.DVISolverConfig:
    """Build a deterministic APGD family-solver configuration."""
    return kamino_config.DVISolverConfig(
        contact_solver="apgd",
        coupling_iterations=coupling_iterations,
        limit_pgs_sweeps=1,
        contact_pgs_sweeps=1,
        post_stabilization_bilateral=post_stabilization_bilateral,
        tolerance=tolerance,
        regularization=1.0e-6,
        apgd=kamino_config.DVIAPGDConfig(
            max_iterations=apgd_max_iterations,
            max_backtrack_iterations=20,
            tolerance=2.0e-5,
            min_iterations=1,
            early_exit=True,
            use_graph_conditionals=False,
        ),
    )


def _build_three_box_stack() -> newton.ModelBuilder:
    """Build three touching boxes with redundant four-point face manifolds."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, margin=0.0, gap=0.0)
    half_extent = 0.1
    mass = 1.0
    body_inertia = inertia.compute_inertia_box_from_mass(
        mass=mass,
        hx=half_extent,
        hy=half_extent,
        hz=half_extent,
    )
    for box_index in range(3):
        body = builder.add_body(
            label=f"box_{box_index}",
            xform=wp.transformf(
                (0.0, 0.0, half_extent + 2.0 * half_extent * box_index),
                wp.quat_identity(),
            ),
            mass=mass,
            inertia=body_inertia,
            lock_inertia=True,
        )
        builder.add_shape_box(
            label=f"box_{box_index}_shape",
            body=body,
            hx=half_extent,
            hy=half_extent,
            hz=half_extent,
            cfg=shape_cfg,
        )
    builder.add_ground_plane(cfg=shape_cfg)
    return builder


def _make_problem(
    device: wp.DeviceLike,
    builder: newton.ModelBuilder,
    *,
    sparse: bool,
    max_world_contacts: int,
    dt: float,
    constraints: kamino_config.ConstraintStabilizationConfig | None = None,
) -> tuple[ModelKamino, object, object, object, object, DualProblem]:
    """Build the Kamino containers and a preconditioned dual problem."""
    model = ModelKamino.from_newton(builder.finalize(device=device))
    model, data, state, limits, detector, jacobians = make_containers(
        model=model,
        max_world_contacts=max_world_contacts,
        sparse=sparse,
        dt=dt,
    )
    update_containers(model, data, state, limits, detector, jacobians)
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=detector.contacts,
        jacobians=jacobians,
        solver=None if sparse else LLTBlockedSolver,
        sparse=sparse,
        config=DualProblem.Config(
            constraints=constraints or kamino_config.ConstraintStabilizationConfig(),
            dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=True),
        ),
    )
    problem.build(model, data, jacobians, limits, detector.contacts)
    return model, data, limits, detector.contacts, jacobians, problem


def _solve_problem(
    model: ModelKamino,
    data: object,
    limits: object,
    contacts: object,
    jacobians: object,
    problem: DualProblem,
    config: kamino_config.DVISolverConfig,
) -> DVISolver:
    """Cold-start and solve one fully built dual problem."""
    solver = DVISolver(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        problem=problem,
        config=config,
        warmstart=WarmStartMode.NONE,
    )
    solver.coldstart()
    solver.solve(problem)
    return solver


def _assert_preconditioned_objective(
    test: unittest.TestCase,
    model: ModelKamino,
    problem: DualProblem,
    solver: DVISolver,
) -> None:
    """Match the associated CCP objective in physical coordinates."""
    metrics = SolutionMetrics(model=model)
    metrics.reset()
    metrics._evaluate_dual_problem_perf(
        solver.data.state.sigma,
        solver.data.solution.lambdas,
        solver.data.solution.v_plus,
        problem,
        contact_law="associated_at",
    )

    lambdas = extract_problem_vector(
        problem.delassus,
        solver.data.solution.lambdas.numpy(),
        only_active_dims=True,
    )
    physical_velocity = extract_problem_vector(
        problem.delassus,
        solver.data.solution.v_plus.numpy(),
        only_active_dims=True,
    )
    physical_compliance = extract_problem_vector(
        problem.delassus,
        problem.data.E.numpy(),
        only_active_dims=True,
    )
    represented_free_velocity = extract_problem_vector(
        problem.delassus,
        problem.data.v_f.numpy(),
        only_active_dims=True,
    )
    preconditioner = extract_problem_vector(
        problem.delassus,
        problem.data.P.numpy(),
        only_active_dims=True,
    )
    test.assertTrue(any(np.any(np.abs(scale - 1.0) > 1.0e-3) for scale in preconditioner))
    expected = np.asarray(
        [
            0.5
            * np.dot(
                impulse,
                velocity + compliance * impulse + represented_free / scale,
            )
            for impulse, velocity, compliance, represented_free, scale in zip(
                lambdas,
                physical_velocity,
                physical_compliance,
                represented_free_velocity,
                preconditioner,
                strict=True,
            )
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(metrics.data.f_ccp.numpy(), expected, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(metrics.data.f_ncp.numpy(), expected, rtol=2.0e-5, atol=2.0e-6)
    test.assertTrue(np.all(np.isfinite(expected)))


def test_redundant_stack_support_and_preconditioned_objective(
    test: unittest.TestCase,
    device: wp.DeviceLike,
) -> None:
    """Support a redundant box stack and match its physical APGD objective."""
    results: dict[bool, np.ndarray] = {}
    expected_total_normal_impulse = 6.0 * 9.81e-3
    for sparse in (False, True):
        with test.subTest(sparse=sparse):
            model, data, limits, contacts, jacobians, problem = _make_problem(
                device,
                _build_three_box_stack(),
                sparse=sparse,
                max_world_contacts=32,
                dt=1.0e-3,
            )
            test.assertEqual(int(problem.data.nc.numpy()[0]), 12)
            solver = _solve_problem(
                model,
                data,
                limits,
                contacts,
                jacobians,
                problem,
                _apgd_config(coupling_iterations=24, apgd_max_iterations=400),
            )
            status = solver.data.status.numpy()[0]
            test.assertEqual(int(status["converged"]), 1, msg=str(status))
            test.assertEqual(int(status["iterations"]), 1)
            test.assertGreater(int(status["contact_iterations"]), 1)
            test.assertLessEqual(float(status["contact_solver_residual"]), 2.0e-5)
            for residual_name in ("r_natural", "r_p", "r_d", "r_c", "r_b"):
                test.assertLessEqual(float(status[residual_name]), 2.0e-4, msg=str(status))
            test.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))

            vio = int(problem.data.vio.numpy()[0])
            ccgo = int(problem.data.ccgo.numpy()[0])
            count = int(problem.data.nc.numpy()[0])
            contact_impulses = solver.data.solution.lambdas.numpy()[vio + ccgo : vio + ccgo + 3 * count].reshape(-1, 3)
            test.assertAlmostEqual(
                float(np.sum(contact_impulses[:, 2])),
                expected_total_normal_impulse,
                delta=0.03 * expected_total_normal_impulse,
            )
            _assert_preconditioned_objective(test, model, problem, solver)
            results[sparse] = contact_impulses

    if False in results and True in results:
        np.testing.assert_allclose(results[False], results[True], rtol=2.0e-3, atol=2.0e-5)


def _build_articulated_contact_limit() -> newton.ModelBuilder:
    """Build one body coupled through a bilateral joint, limit, and contact."""
    builder = newton.ModelBuilder(gravity=wp.vec3f(0.0, 0.0, -9.81))
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0
    body_xform = wp.transformf((0.4, 0.0, 0.45), wp.quat_identity())
    body = builder.add_link(mass=1.0, xform=body_xform)
    builder.add_shape_sphere(body=body, radius=0.5)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Y,
        parent_xform=body_xform,
        child_xform=wp.transform_identity(),
        limit_lower=0.1,
        limit_upper=1.0,
    )
    builder.add_articulation([joint])
    builder.add_ground_plane()
    return builder


def test_articulated_contact_limit_runs_multiple_apgd_family_sweeps(
    test: unittest.TestCase,
    device: wp.DeviceLike,
) -> None:
    """Couple active limit, bilateral, and APGD contact rows over repeated sweeps."""
    results: dict[tuple[bool, bool], np.ndarray] = {}
    coupling_iterations = 12
    for sparse in (False, True):
        for post_stabilization_bilateral in (False, True):
            with test.subTest(sparse=sparse, post_stabilization_bilateral=post_stabilization_bilateral):
                model, data, limits, contacts, jacobians, problem = _make_problem(
                    device,
                    _build_articulated_contact_limit(),
                    sparse=sparse,
                    max_world_contacts=2,
                    dt=0.01,
                    constraints=kamino_config.ConstraintStabilizationConfig(
                        contact_compliance=1.0e-3,
                        contact_stabilization_time=0.02,
                    ),
                )
                test.assertEqual(int(problem.data.njc.numpy()[0]), 5)
                test.assertEqual(int(problem.data.nl.numpy()[0]), 1)
                test.assertEqual(int(problem.data.nc.numpy()[0]), 1)
                solver = _solve_problem(
                    model,
                    data,
                    limits,
                    contacts,
                    jacobians,
                    problem,
                    _apgd_config(
                        coupling_iterations=coupling_iterations,
                        post_stabilization_bilateral=post_stabilization_bilateral,
                    ),
                )
                status = solver.data.status.numpy()[0]
                test.assertEqual(int(status["iterations"]), coupling_iterations)
                test.assertGreaterEqual(int(status["contact_iterations"]), coupling_iterations)
                test.assertEqual(int(status["limit_iterations"]), coupling_iterations)
                test.assertEqual(int(status["converged"]), 1, msg=str(status))
                for residual_name in ("r_natural", "r_p", "r_d", "r_c", "r_b"):
                    test.assertLessEqual(float(status[residual_name]), 2.0e-4, msg=str(status))
                solution = solver.data.solution.lambdas.numpy().copy()
                test.assertTrue(np.all(np.isfinite(solution)))
                results[sparse, post_stabilization_bilateral] = solution

    for post_stabilization_bilateral in (False, True):
        dense_key = (False, post_stabilization_bilateral)
        sparse_key = (True, post_stabilization_bilateral)
        if dense_key in results and sparse_key in results:
            np.testing.assert_allclose(results[dense_key], results[sparse_key], rtol=3.0e-4, atol=3.0e-5)


def _run_compliant_sphere(
    device: wp.DeviceLike,
    *,
    sparse: bool,
    dt: float,
    duration: float,
    compliance: float,
) -> tuple[float, np.ndarray]:
    """Run a public APGD sphere rollout and return its signed surface height."""
    radius = 0.1
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverKamino.register_custom_attributes(builder)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0, gap=0.0)
    body = builder.add_body(
        xform=wp.transformf((0.0, 0.0, radius), wp.quat_identity()),
        mass=1.0,
        inertia=inertia.compute_inertia_sphere_from_mass(mass=1.0, radius=radius),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body=body, radius=radius, cfg=shape_cfg)
    builder.add_ground_plane(cfg=shape_cfg)
    model = builder.finalize(device=device)
    config = SolverKamino.Config(
        dynamics_solver="dvi",
        use_collision_detector=True,
        sparse_dynamics=sparse,
        sparse_jacobian=sparse,
        constraints=kamino_config.ConstraintStabilizationConfig(
            delta=0.0,
            contact_compliance=compliance,
            contact_stabilization_time=0.01,
            contact_recovery_speed=1.0,
        ),
        dynamics=kamino_config.ConstrainedDynamicsConfig(
            preconditioning=False,
            linear_solver_type="CR" if sparse else "LLTB",
        ),
        dvi=_apgd_config(coupling_iterations=4, apgd_max_iterations=8),
    )
    solver = SolverKamino(model, config=config)
    state_0 = model.state()
    state_1 = model.state()
    for _ in range(round(duration / dt)):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    status = solver.status.numpy()[0]
    if int(status["converged"]) != 1:
        raise AssertionError(f"APGD compliant rollout did not converge: {status}")
    if any(float(status[name]) > 2.0e-4 for name in ("r_natural", "r_p", "r_d", "r_c", "r_b")):
        raise AssertionError(f"APGD compliant rollout exceeded its DVI tolerance: {status}")
    pose = state_0.body_q.numpy()[body]
    velocity = state_0.body_qd.numpy()[body]
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(velocity)):
        raise AssertionError("APGD compliant rollout produced a non-finite state.")
    return float(pose[2] - radius), velocity


def test_public_compliant_apgd_is_timestep_consistent(
    test: unittest.TestCase,
    device: wp.DeviceLike,
) -> None:
    """Approach the same compliant equilibrium at two simulation time steps."""
    compliance = 1.0e-4
    duration = 0.06
    expected_penetration = -compliance * 9.81
    results: dict[tuple[bool, float], tuple[float, np.ndarray]] = {}
    for sparse in (False, True):
        for dt in (0.004, 0.002):
            with test.subTest(sparse=sparse, dt=dt):
                signed_height, velocity = _run_compliant_sphere(
                    device,
                    sparse=sparse,
                    dt=dt,
                    duration=duration,
                    compliance=compliance,
                )
                test.assertAlmostEqual(signed_height, expected_penetration, delta=1.5e-4)
                test.assertLess(abs(float(velocity[2])), 5.0e-3)
                results[sparse, dt] = (signed_height, velocity)

        if (sparse, 0.004) in results and (sparse, 0.002) in results:
            test.assertAlmostEqual(results[sparse, 0.004][0], results[sparse, 0.002][0], delta=8.0e-5)

    if (False, 0.002) in results and (True, 0.002) in results:
        np.testing.assert_allclose(results[False, 0.002][1], results[True, 0.002][1], rtol=0.0, atol=2.0e-5)


def test_dr_legs_sparse_apgd_graph_smoke(
    test: unittest.TestCase,
    device: wp.DeviceLike,
) -> None:
    """Run the real DR Legs contact scene through sparse APGD graph replay.

    This is deliberately a finite-work smoke test rather than a coupled-family
    convergence gate. DR Legs' example-tuned four-sweep budget leaves a split
    residual after first contact; the smaller articulated test above provides
    the strict repeated ``L -> B -> C`` convergence regression with optional
    post-stabilization bilateral solves.
    """
    if not device.is_cuda:
        test.skipTest("DR Legs APGD graph replay requires CUDA.")
    if not wp.is_conditional_graph_supported():
        test.skipTest("DR Legs APGD needs nested conditional graph support.")

    from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
    from newton.viewer import ViewerNull  # noqa: PLC0415

    solver_init = SolverKamino.__init__

    def init_apgd_solver(self, model, config=None):
        """Select APGD after the example applies its DVI-specific defaults."""
        config.dvi.contact_solver = "apgd"
        config.dvi.contact_law = None
        config.dvi.apgd = kamino_config.DVIAPGDConfig(
            max_iterations=30,
            max_backtrack_iterations=10,
            tolerance=5.0e-5,
            min_iterations=1,
            early_exit=True,
            use_graph_conditionals=True,
        )
        solver_init(self, model, config=config)

    args = SimpleNamespace(
        world_count=1,
        use_kamino_contacts=True,
        dynamics_solver="dvi",
        joint_effort_limit=np.inf,
    )
    with mock.patch.object(SolverKamino, "__init__", new=init_apgd_solver):
        example = Example(ViewerNull(num_frames=1), args)

    test.assertTrue(example.config.sparse_dynamics)
    test.assertEqual(example.config.dvi.contact_solver, "apgd")
    test.assertEqual(example.config.dvi.resolved_contact_law, "associated_at")
    contact_seen = False
    contact_work_seen = False
    for _ in range(12):
        example.step()
        contact_count = int(example.solver._contacts_kamino.world_active_contacts.numpy()[0])
        status = example.solver.status.numpy()[0]
        contact_seen |= contact_count > 0
        contact_work_seen |= int(status["contact_iterations"]) > 0
        test.assertEqual(int(status["invalid_contact_preconditioner"]), 0, msg=str(status))
        test.assertTrue(np.isfinite(float(status["r_natural"])), msg=str(status))
        test.assertTrue(np.isfinite(float(status["contact_solver_residual"])), msg=str(status))
        test.assertLess(float(status["r_natural"]), 0.5, msg=str(status))
        test.assertLess(float(status["contact_solver_residual"]), 5.0e-4, msg=str(status))
        test.assertTrue(np.all(np.isfinite(example.state_0.body_q.numpy())))
        test.assertTrue(np.all(np.isfinite(example.state_0.body_qd.numpy())))
        test.assertTrue(np.all(np.isfinite(example.solver._solver_kamino.solver_fd.data.solution.lambdas.numpy())))
        test.assertLess(float(np.max(np.abs(example.state_0.body_qd.numpy()))), 100.0)

    test.assertTrue(contact_seen)
    test.assertTrue(contact_work_seen)


def test_dr_legs_sparse_apgd_terminal_residual(
    test: unittest.TestCase,
    device: wp.DeviceLike,
) -> None:
    """Converge the full DR Legs DVI system after contact settles."""
    if not device.is_cuda:
        test.skipTest("DR Legs APGD terminal convergence requires CUDA.")
    if not wp.is_conditional_graph_supported():
        test.skipTest("DR Legs APGD needs nested conditional graph support.")

    from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
    from newton.viewer import ViewerNull  # noqa: PLC0415

    terminal_tolerance = 1.0e-3
    solver_init = SolverKamino.__init__

    def init_apgd_solver(self, model, config=None):
        """Give the coupled-family fixed point a convergence-test budget."""
        config.compute_solution_metrics = True
        config.dvi.contact_solver = "apgd"
        config.dvi.contact_law = None
        config.dvi.coupling_iterations = 128
        config.dvi.post_stabilization_bilateral = True
        config.dvi.tolerance = terminal_tolerance
        config.dvi.apgd = kamino_config.DVIAPGDConfig(
            max_iterations=20,
            max_backtrack_iterations=20,
            tolerance=terminal_tolerance,
            min_iterations=1,
            early_exit=True,
            use_graph_conditionals=True,
        )
        solver_init(self, model, config=config)

    args = SimpleNamespace(
        world_count=1,
        use_kamino_contacts=True,
        dynamics_solver="dvi",
        joint_effort_limit=np.inf,
    )
    with mock.patch.object(SolverKamino, "__init__", new=init_apgd_solver):
        example = Example(ViewerNull(num_frames=1), args)

    test.assertTrue(example.config.sparse_dynamics)
    test.assertTrue(example.config.sparse_jacobian)
    test.assertTrue(example.config.compute_solution_metrics)
    test.assertEqual(example.config.dvi.contact_solver, "apgd")
    test.assertEqual(example.config.dvi.resolved_contact_law, "associated_at")
    test.assertTrue(example.config.dvi.post_stabilization_bilateral)
    metrics = example.solver._solver_kamino.metrics
    test.assertIsNotNone(metrics)

    required_converged_contact_frames = 4
    consecutive_converged_contact_frames = 0
    contact_seen = False
    last_diagnostics: dict[str, object] = {}
    for _ in range(24):
        example.step()
        contact_count = int(example.solver._contacts_kamino.world_active_contacts.numpy()[0])
        status = example.solver.status.numpy()[0]
        if contact_count == 0:
            consecutive_converged_contact_frames = 0
            continue

        contact_seen = True
        status_residuals = {name: float(status[name]) for name in ("r_natural", "r_p", "r_d", "r_c", "r_b")}
        metric_residuals = {
            name: float(getattr(metrics.data, name).numpy()[0])
            for name in ("r_ncp_primal", "r_ncp_dual", "r_ncp_compl", "r_vi_natmap")
        }
        contact_solver_residual = float(status["contact_solver_residual"])
        last_diagnostics = {
            "status": status,
            "status_residuals": status_residuals,
            "metric_residuals": metric_residuals,
            "contact_solver_residual": contact_solver_residual,
        }
        terminal_residuals_converged = all(
            value <= terminal_tolerance for value in (*status_residuals.values(), *metric_residuals.values())
        )
        contact_phase_converged = (
            int(status["contact_iterations"]) > 0 and contact_solver_residual <= terminal_tolerance
        )
        if int(status["converged"]) == 1 and terminal_residuals_converged and contact_phase_converged:
            consecutive_converged_contact_frames += 1
            if consecutive_converged_contact_frames == required_converged_contact_frames:
                break
        else:
            consecutive_converged_contact_frames = 0

    test.assertTrue(contact_seen)
    test.assertEqual(
        consecutive_converged_contact_frames,
        required_converged_contact_frames,
        msg=f"DR Legs APGD did not maintain its terminal tolerance; final diagnostics: {last_diagnostics}",
    )


_DEVICES = get_test_devices(mode="basic")

add_function_test(
    TestDVIAPGDDynamics,
    "test_redundant_stack_support_and_preconditioned_objective",
    test_redundant_stack_support_and_preconditioned_objective,
    devices=_DEVICES,
)
add_function_test(
    TestDVIAPGDDynamics,
    "test_articulated_contact_limit_runs_multiple_apgd_family_sweeps",
    test_articulated_contact_limit_runs_multiple_apgd_family_sweeps,
    devices=_DEVICES,
)
add_function_test(
    TestDVIAPGDDynamics,
    "test_public_compliant_apgd_is_timestep_consistent",
    test_public_compliant_apgd_is_timestep_consistent,
    devices=_DEVICES,
)
add_function_test(
    TestDVIAPGDDynamics,
    "test_dr_legs_sparse_apgd_graph_smoke",
    test_dr_legs_sparse_apgd_graph_smoke,
    devices=_DEVICES,
)
add_function_test(
    TestDVIAPGDDynamics,
    "test_dr_legs_sparse_apgd_terminal_residual",
    test_dr_legs_sparse_apgd_terminal_residual,
    devices=_DEVICES,
)


if __name__ == "__main__":
    unittest.main()
