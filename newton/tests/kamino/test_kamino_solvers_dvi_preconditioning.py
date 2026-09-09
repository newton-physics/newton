# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for physical DVI status with dual preconditioning."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.dvi.kernels import (
    _compute_dvi_status_residuals,
    _initialize_dvi_status,
    _unprecondition_dvi_solution,
)
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct, DVIStatus, convert_config_to_struct
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.unittest_utils import add_function_test, get_test_devices
from newton.tests.utils import basics


class TestDVIPreconditioning(unittest.TestCase):
    """Validate DVI preconditioning through kernels and the public solver."""

    pass


def _array(device: wp.DeviceLike, values: list[float] | list[int], dtype: type) -> wp.array:
    """Construct a one-dimensional Warp test array."""
    return wp.array(values, dtype=dtype, device=device)


def _evaluate_synthetic_status(
    device: wp.DeviceLike,
    preconditioner: np.ndarray,
) -> tuple[np.void, np.ndarray, np.ndarray, np.ndarray]:
    """Unscale a synthetic iterate and evaluate its terminal DVI status."""
    physical_lambdas = np.array([0.0, 1.25, 0.1, 0.1, 0.0, 0.4], dtype=np.float32)
    physical_v_aug = np.array([0.03, -0.2, -0.3, 0.0, 0.0, -0.1], dtype=np.float32)
    physical_s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.07], dtype=np.float32)
    represented_lambdas = physical_lambdas / preconditioner
    represented_v_aug = preconditioner * physical_v_aug
    represented_s = preconditioner * physical_s

    lambdas = _array(device, represented_lambdas.tolist(), wp.float32)
    v_aug = _array(device, represented_v_aug.tolist(), wp.float32)
    s = _array(device, represented_s.tolist(), wp.float32)
    v_plus = _array(device, represented_v_aug.tolist(), wp.float32)
    P = _array(device, preconditioner.tolist(), wp.float32)
    status = wp.zeros(1, dtype=DVIStatus, device=device)
    config = wp.array(
        [convert_config_to_struct(kamino_config.DVISolverConfig(tolerance=0.0))],
        dtype=DVIConfigStruct,
        device=device,
    )

    wp.launch(_initialize_dvi_status, dim=1, inputs=[config, status], device=device)
    wp.launch(
        _unprecondition_dvi_solution,
        dim=(1, 6),
        inputs=[
            _array(device, [6], wp.int32),
            _array(device, [0], wp.int32),
            P,
            status,
            s,
            v_aug,
            lambdas,
            v_plus,
        ],
        device=device,
    )
    wp.launch(
        _compute_dvi_status_residuals,
        dim=1,
        inputs=[
            _array(device, [6], wp.int32),  # dim
            _array(device, [0], wp.int32),  # vio
            _array(device, [1], wp.int32),  # bilateral rows
            _array(device, [1], wp.int32),  # bounded rows
            _array(device, [1], wp.int32),  # limit rows
            _array(device, [1], wp.int32),  # contacts
            _array(device, [1], wp.int32),  # bounded group offset
            _array(device, [2], wp.int32),  # limit group offset
            _array(device, [3], wp.int32),  # contact group offset
            _array(device, [0], wp.int32),  # bounded-vector offset
            _array(device, [0], wp.int32),  # contact-vector offset
            _array(device, [0.5], wp.float32),
            P,
            _array(device, [-1.0 / float(preconditioner[1])], wp.float32),
            _array(device, [1.0 / float(preconditioner[1])], wp.float32),
            config,
            v_aug,
            lambdas,
            status,
        ],
        device=device,
    )
    return status.numpy()[0], lambdas.numpy(), v_aug.numpy(), s.numpy()


def test_terminal_status_uses_physical_coordinates(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Keep terminal residuals invariant under cone-compatible dual scaling."""
    physical_status, physical_lambdas, physical_v_aug, physical_s = _evaluate_synthetic_status(
        device,
        np.ones(6, dtype=np.float32),
    )
    scaled_status, scaled_lambdas, scaled_v_aug, scaled_s = _evaluate_synthetic_status(
        device,
        np.array([0.2, 2.0, 0.5, 0.25, 0.25, 0.25], dtype=np.float32),
    )

    np.testing.assert_allclose(scaled_lambdas, physical_lambdas, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(scaled_v_aug, physical_v_aug, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(scaled_s, physical_s, rtol=0.0, atol=1.0e-7)
    for name in ("r_p", "r_d", "r_c", "r_b", "r_natural"):
        test.assertGreater(float(physical_status[name]), 0.0)
        test.assertAlmostEqual(float(scaled_status[name]), float(physical_status[name]), places=6)
    test.assertAlmostEqual(float(physical_status["r_p"]), 0.25, places=6)
    test.assertEqual(int(physical_status["converged"]), 0)
    test.assertEqual(int(scaled_status["converged"]), 0)


def _run_public_contact_step(
    device: wp.DeviceLike,
    *,
    sparse: bool,
    contact_solver: str,
    preconditioning: bool,
) -> dict[str, object]:
    """Run one deliberately under-converged contact step through SolverKamino."""
    model = basics.build_box_on_plane().finalize(device=device)
    dvi = kamino_config.DVISolverConfig(
        contact_solver=contact_solver,
        max_alternating_iterations=1,
        inequality_sweeps_per_iteration=1,
        tolerance=0.0,
        regularization=1.0e-12,
        warmstart_mode="none",
        apgd=kamino_config.DVIAPGDConfig(
            max_iterations=1,
            max_backtrack_iterations=1,
            tolerance=0.0,
            min_iterations=1,
            early_exit=False,
            use_graph_conditionals=False,
        ),
    )
    config = SolverKamino.Config(
        dynamics_solver="dvi",
        sparse_dynamics=sparse,
        sparse_jacobian=sparse,
        use_collision_detector=True,
        dynamics=kamino_config.ConstrainedDynamicsConfig(
            preconditioning=preconditioning,
            linear_solver_type="CR" if sparse else "LLTB",
        ),
        dvi=dvi,
        compute_solution_metrics=True,
    )
    solver = SolverKamino(model, config=config)
    state_in = model.state()
    state_out = model.state()
    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)

    implementation = solver._solver_kamino
    problem = implementation.problem_fd
    solution = implementation.solver_fd.data.solution
    vio = int(problem.data.vio.numpy()[0])
    dim = int(problem.data.dim.numpy()[0])
    metrics = implementation.metrics.data
    return {
        "status": solver.status.numpy()[0].copy(),
        "metric_residuals": np.array(
            [
                metrics.r_ncp_primal.numpy()[0],
                metrics.r_ncp_dual.numpy()[0],
                metrics.r_ncp_compl.numpy()[0],
                metrics.r_vi_natmap.numpy()[0],
            ]
        ),
        "lambdas": solution.lambdas.numpy()[vio : vio + dim].copy(),
        "v_plus": solution.v_plus.numpy()[vio : vio + dim].copy(),
        "body_q": state_out.body_q.numpy().copy(),
        "body_qd": state_out.body_qd.numpy().copy(),
        "preconditioner": problem.data.P.numpy()[vio : vio + dim].copy(),
        "contact_count": int(problem.data.nc.numpy()[0]),
    }


def test_public_preconditioning_matches_physical_contact_solution(
    test: unittest.TestCase,
    device: wp.DeviceLike,
) -> None:
    """Match physical status and dynamics for public dense/sparse PGS and APGD."""
    for contact_solver in ("pgs", "apgd"):
        for sparse in (False, True):
            with test.subTest(contact_solver=contact_solver, sparse=sparse):
                baseline = _run_public_contact_step(
                    device,
                    sparse=sparse,
                    contact_solver=contact_solver,
                    preconditioning=False,
                )
                scaled = _run_public_contact_step(
                    device,
                    sparse=sparse,
                    contact_solver=contact_solver,
                    preconditioning=True,
                )

                test.assertEqual(baseline["contact_count"], 4)
                test.assertEqual(scaled["contact_count"], 4)
                np.testing.assert_array_equal(baseline["preconditioner"], 1.0)
                test.assertTrue(np.all(np.asarray(scaled["preconditioner"]) != 1.0))
                for name in ("lambdas", "v_plus", "body_q", "body_qd"):
                    np.testing.assert_allclose(scaled[name], baseline[name], rtol=2.0e-5, atol=2.0e-7)

                baseline_status = baseline["status"]
                scaled_status = scaled["status"]
                test.assertEqual(int(baseline_status["converged"]), 0)
                test.assertEqual(int(scaled_status["converged"]), 0)
                test.assertGreater(float(baseline_status["r_d"]), 1.0e-4)
                test.assertGreater(float(baseline_status["r_natural"]), 1.0e-4)
                for name in ("r_p", "r_d", "r_c", "r_b", "r_natural"):
                    test.assertAlmostEqual(float(scaled_status[name]), float(baseline_status[name]), places=6)

                for index, name in enumerate(("r_p", "r_d", "r_c", "r_natural")):
                    test.assertAlmostEqual(
                        float(baseline_status[name]),
                        float(np.asarray(baseline["metric_residuals"])[index]),
                        places=6,
                    )
                    test.assertAlmostEqual(
                        float(scaled_status[name]),
                        float(np.asarray(scaled["metric_residuals"])[index]),
                        places=6,
                    )


def _build_heterogeneous_preconditioning_problem(
    device: wp.DeviceLike,
    *,
    sparse: bool,
) -> DualProblem:
    """Build two compliant contact worlds with different preconditioning settings."""
    builder = newton.ModelBuilder()
    basics.build_sphere_on_plane(builder=builder, friction=0.0, restitution=0.0)
    basics.build_sphere_on_plane(builder=builder, friction=0.0, restitution=0.0)
    model = ModelKamino.from_newton(builder.finalize(device=device))
    model, data, state, limits, detector, jacobians = make_containers(
        model=model,
        max_world_contacts=1,
        sparse=sparse,
        dt=0.01,
    )
    update_containers(model, data, state, limits, detector, jacobians)

    configs = [
        DualProblem.Config(
            constraints=kamino_config.ConstraintStabilizationConfig(
                contact_compliance=2.0e-4,
                contact_stabilization_time=0.02,
            ),
            dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=enabled),
        )
        for enabled in (True, False)
    ]
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=detector.contacts,
        jacobians=jacobians,
        solver=None if sparse else LLTBlockedSolver,
        config=configs,
        sparse=sparse,
    )
    problem.build(model, data, jacobians, limits, detector.contacts)
    return problem


def test_sparse_heterogeneous_world_preconditioning_preserves_identity(
    test: unittest.TestCase,
    device: wp.DeviceLike,
) -> None:
    """Preserve identity scaling in sparse worlds that disable preconditioning."""
    results: dict[bool, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}
    for sparse in (False, True):
        with test.subTest(sparse=sparse):
            problem = _build_heterogeneous_preconditioning_problem(device, sparse=sparse)
            np.testing.assert_array_equal(problem.data.nc.numpy(), np.array([1, 1], dtype=np.int32))

            dimensions = problem.data.dim.numpy()
            vector_offsets = problem.data.vio.numpy()
            packed_P = problem.data.P.numpy()
            packed_E = problem.data.E.numpy()
            packed_E_hat = problem.data.E_hat.numpy()
            world_P: list[np.ndarray] = []
            world_E: list[np.ndarray] = []
            world_E_hat: list[np.ndarray] = []
            for world in range(2):
                world_slice = slice(int(vector_offsets[world]), int(vector_offsets[world] + dimensions[world]))
                world_P.append(packed_P[world_slice].copy())
                world_E.append(packed_E[world_slice].copy())
                world_E_hat.append(packed_E_hat[world_slice].copy())

            test.assertTrue(np.any(np.abs(world_P[0] - 1.0) > 1.0e-4))
            test.assertTrue(np.all(world_E[0] > 0.0))
            test.assertTrue(np.all(world_E[1] > 0.0))
            np.testing.assert_allclose(world_E_hat[0], world_P[0] * world_P[0] * world_E[0], rtol=1.0e-6)
            np.testing.assert_array_equal(world_P[1], np.ones_like(world_P[1]))
            np.testing.assert_array_equal(world_E_hat[1], world_E[1])

            if sparse:
                raw_diagonal = wp.empty_like(problem.data.P)
                problem.delassus.diagonal(raw_diagonal)
                disabled_slice = slice(int(vector_offsets[1]), int(vector_offsets[1] + dimensions[1]))
                test.assertTrue(np.any(np.abs(raw_diagonal.numpy()[disabled_slice] - 1.0) > 1.0e-4))

            results[sparse] = (world_P, world_E, world_E_hat)

    for field in range(3):
        for world in range(2):
            np.testing.assert_allclose(results[False][field][world], results[True][field][world], rtol=1.0e-6)


_DEVICES = get_test_devices(mode="basic")

add_function_test(
    TestDVIPreconditioning,
    "test_terminal_status_uses_physical_coordinates",
    test_terminal_status_uses_physical_coordinates,
    devices=_DEVICES,
)
add_function_test(
    TestDVIPreconditioning,
    "test_public_preconditioning_matches_physical_contact_solution",
    test_public_preconditioning_matches_physical_contact_solution,
    devices=_DEVICES,
)
add_function_test(
    TestDVIPreconditioning,
    "test_sparse_heterogeneous_world_preconditioning_preserves_identity",
    test_sparse_heterogeneous_world_preconditioning_preserves_identity,
    devices=_DEVICES,
)


if __name__ == "__main__":
    unittest.main()
