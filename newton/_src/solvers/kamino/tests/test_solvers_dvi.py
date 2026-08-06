# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Kamino DVI solver."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core import ModelBuilderKamino, inertia
from newton._src.solvers.kamino._src.core.shapes import BoxShape, SphereShape
from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.integrators.euler import integrate_euler_semi_implicit
from newton._src.solvers.kamino._src.kinematics.constraints import unpack_constraint_solutions, update_constraints_info
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino._src.linalg import LLTBlockedRCMSolver, LLTBlockedSolver
from newton._src.solvers.kamino._src.models.builders import basics, testing
from newton._src.solvers.kamino._src.models.builders import utils as builder_utils
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi.kernels import _initialize_dvi_status
from newton._src.solvers.kamino._src.solvers.dvi.sparse import (
    _SPARSE_DELASSUS_ROWS_JOINTS,
    _SPARSE_DELASSUS_ROWS_UNILATERAL,
    _sparse_delassus_matvec_rows,
)
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _color_mapped_dvi_inequalities,
    _solve_dvi_sparse_inequalities_pgs,
)
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct, convert_config_to_struct
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_padmm import TestSetup
from newton._src.solvers.kamino.tests.utils.extract import extract_delassus, extract_problem_vector
from newton._src.solvers.kamino.tests.utils.make import make_containers, make_test_problem_fourbar, update_containers
from newton.tests.utils import basics as public_basics


def _build_five_box_stack() -> ModelBuilderKamino:
    """Build a vertical stack with four-point contacts at every interface."""
    builder = ModelBuilderKamino(default_world=False)
    world = builder.add_world(name="five_box_stack")
    for box_index in range(5):
        body = builder.add_rigid_body(
            name=f"box_{box_index}",
            m_i=1.0,
            i_I_i=inertia.solid_cuboid_body_moment_of_inertia(1.0, 0.2, 0.2, 0.2),
            q_i_0=wp.transformf(0.0, 0.0, 0.1 + 0.2 * box_index, 0.0, 0.0, 0.0, 1.0),
            u_i_0=wp.spatial_vectorf(0.0),
            world_index=world,
        )
        builder.add_geometry(body=body, shape=BoxShape(0.1, 0.1, 0.1), world_index=world)
    builder.add_geometry(
        body=-1,
        shape=BoxShape(10.0, 10.0, 0.5),
        offset=wp.transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
        world_index=world,
    )
    return builder


def _build_high_mass_ratio_sphere_stack() -> ModelBuilderKamino:
    """Build a two-sphere stack with a 100:1 mass ratio."""
    builder = ModelBuilderKamino(default_world=False)
    world = builder.add_world(name="high_mass_ratio_sphere_stack")
    for body_index, mass in enumerate((1.0, 100.0)):
        body = builder.add_rigid_body(
            name=f"sphere_{body_index}",
            m_i=mass,
            i_I_i=inertia.solid_sphere_body_moment_of_inertia(mass, 0.1),
            q_i_0=wp.transformf(0.0, 0.0, 0.1 + 0.2 * body_index, 0.0, 0.0, 0.0, 1.0),
            u_i_0=wp.spatial_vectorf(0.0),
            world_index=world,
        )
        builder.add_geometry(body=body, shape=SphereShape(0.1), world_index=world)
    builder.add_geometry(
        body=-1,
        shape=BoxShape(10.0, 10.0, 0.5),
        offset=wp.transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
        world_index=world,
    )
    return builder


def _build_reduced_kapla_tower(layer_count: int = 6) -> tuple[newton.ModelBuilder, list[int], np.ndarray]:
    """Build a compact pinwheel tower for dynamic contact regressions."""
    plank_length = 0.30
    plank_thickness = 0.02
    plank_height = 0.06
    half_side = 0.5 * (plank_length + plank_thickness)
    local_specs = (
        ((0.5 * plank_length, 0.5 * plank_thickness), 0.0),
        ((plank_length + 0.5 * plank_thickness, 0.5 * plank_length), 0.5 * np.pi),
        ((0.5 * plank_length + plank_thickness, plank_length + 0.5 * plank_thickness), np.pi),
        ((0.5 * plank_thickness, 0.5 * plank_length + plank_thickness), -0.5 * np.pi),
    )

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverKamino.register_custom_attributes(builder)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=0.0, margin=0.0)
    bodies = []
    initial_positions = []
    for layer in range(layer_count):
        layer_yaw = 0.25 * np.pi if layer % 2 else 0.0
        cos_layer = float(np.cos(layer_yaw))
        sin_layer = float(np.sin(layer_yaw))
        z = 0.5 * plank_height + layer * plank_height
        for (local_x, local_y), local_yaw in local_specs:
            centered_x = local_x - half_side
            centered_y = local_y - half_side
            x = cos_layer * centered_x - sin_layer * centered_y
            y = sin_layer * centered_x + cos_layer * centered_y
            yaw = float(layer_yaw + local_yaw)
            body = builder.add_body(
                xform=wp.transformf(
                    (x, y, z),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
                ),
            )
            builder.add_shape_box(
                body=body,
                hx=0.5 * plank_length,
                hy=0.5 * plank_thickness,
                hz=0.5 * plank_height,
                cfg=shape_cfg,
            )
            bodies.append(body)
            initial_positions.append((x, y, z))
    builder.add_ground_plane(cfg=shape_cfg)
    return builder, bodies, np.asarray(initial_positions, dtype=np.float32)


def _reduce_solver_status(status: np.ndarray) -> dict[str, object]:
    """Reduce per-world status while requiring every world to converge."""
    return {
        name: bool(np.all(status[name])) if name == "converged" else np.max(status[name]).item()
        for name in status.dtype.names
    }


def _check_solution_matches_dual_problem(testcase: unittest.TestCase, problem: DualProblem, solver: DVISolver):
    """Check that final physical solution vectors match ``D lambda + v_f``."""
    D_np = extract_delassus(problem.delassus, only_active_dims=True)
    v_f_np = extract_problem_vector(problem.delassus, problem.data.v_f.numpy(), only_active_dims=True)
    P_np = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=True)
    lambdas_np = extract_problem_vector(problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True)
    v_plus_np = extract_problem_vector(problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=True)

    status = solver.data.status.numpy()
    for wid in range(problem.data.num_worlds):
        P_inv = np.diag(np.reciprocal(P_np[wid]))
        D_true = P_inv @ D_np[wid] @ P_inv
        v_f_true = P_inv @ v_f_np[wid]
        v_plus_true = D_true @ lambdas_np[wid] + v_f_true
        np.testing.assert_allclose(v_plus_np[wid], v_plus_true, rtol=1e-4, atol=1e-4)

        testcase.assertEqual(int(status[wid]["converged"]), 1, msg=str(status[wid]))
        testcase.assertLessEqual(int(status[wid]["iterations"]), _status_iteration_budget(solver, wid))
        testcase.assertLessEqual(float(status[wid]["r_p"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_d"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_c"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_b"]), solver.config[wid].tolerance)


def _make_dense_dual_problem(model, data, limits, contacts, jacobians) -> DualProblem:
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        solver=LLTBlockedSolver,
        sparse=False,
    )
    problem.build(model=model, data=data, limits=limits, contacts=contacts, jacobians=jacobians)
    return problem


def _make_sparse_dual_problem(model, data, limits, contacts, jacobians) -> DualProblem:
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        sparse=True,
    )
    problem.build(model=model, data=data, limits=limits, contacts=contacts, jacobians=jacobians)
    return problem


def _solve_dvi(
    model,
    problem,
    warmstart: WarmStartMode = WarmStartMode.NONE,
    config: kamino_config.DVISolverConfig | None = None,
    setup: TestSetup | None = None,
) -> DVISolver:
    solver = DVISolver(
        model=model,
        data=setup.data if setup is not None else None,
        limits=setup.limits if setup is not None else None,
        contacts=setup.contacts if setup is not None else None,
        jacobians=setup.jacobians if setup is not None else None,
        config=config
        or kamino_config.DVISolverConfig(
            max_alternating_iterations=300,
            inequality_sweeps_per_iteration=1,
            tolerance=1e-4,
            regularization=1e-5,
        ),
        warmstart=warmstart,
    )
    solver.reset()
    solver.coldstart()
    solver.solve(problem)
    return solver


def _status_iteration_budget(solver: DVISolver, wid: int) -> int:
    config = solver.config[wid]
    return config.max_alternating_iterations * config.inequality_sweeps_per_iteration


def _assert_solver_status_converged(testcase: unittest.TestCase, solver: DVISolver):
    status = solver.data.status.numpy()
    for wid in range(solver.size.num_worlds):
        testcase.assertEqual(int(status[wid]["converged"]), 1, msg=str(status[wid]))
        testcase.assertLessEqual(int(status[wid]["iterations"]), _status_iteration_budget(solver, wid))
        testcase.assertLessEqual(float(status[wid]["r_p"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_d"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_c"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_b"]), solver.config[wid].tolerance)


def _assert_solution_finite(testcase: unittest.TestCase, solver: DVISolver):
    testcase.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
    testcase.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))


def _evaluate_solution_metrics(test: TestSetup, solver: DVISolver) -> dict[str, float]:
    integrate_euler_semi_implicit(model=test.model, data=test.data)
    metrics = SolutionMetrics(model=test.model)
    metrics.evaluate(
        sigma=solver.data.state.sigma,
        lambdas=solver.data.solution.lambdas,
        v_plus=solver.data.solution.v_plus,
        model=test.model,
        data=test.data,
        state_p=test.state_p,
        problem=test.problem,
        jacobians=test.jacobians,
        limits=test.limits,
        contacts=test.contacts,
    )
    return {
        name: float(np.max(getattr(metrics.data, name).numpy()))
        for name in (
            "r_eom",
            "r_kinematics",
            "r_cts_joints",
            "r_cts_limits",
            "r_cts_contacts",
            "r_v_plus",
            "r_ncp_primal",
            "r_ncp_dual",
            "r_ncp_compl",
            "r_vi_natmap",
        )
    }


class TestDVISolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_00_config_selection(self):
        default_config = SolverKamino.Config(dynamics_solver="dvi")
        self.assertFalse(default_config.sparse_dynamics)
        self.assertTrue(default_config.sparse_jacobian)
        self.assertEqual(default_config.integrator, "euler")
        self.assertEqual(default_config.dynamics.linear_solver_type, "LLTBRCM")
        self.assertEqual(default_config.dynamics.linear_solver_kwargs, {})
        self.assertEqual(default_config.dvi.omega, 1.0)
        self.assertEqual(default_config.dvi.max_alternating_iterations, 20)
        self.assertEqual(default_config.dvi.inequality_sweeps_per_iteration, 1)
        self.assertEqual(default_config.dvi.bilateral_solve_interval, 1)
        self.assertEqual(default_config.dvi.bilateral_solver_type, "LLTB")
        self.assertEqual(default_config.dvi.bilateral_solver_kwargs, {})

        dense_config = SolverKamino.Config(
            dynamics_solver="dvi",
            sparse_dynamics=False,
            sparse_jacobian=False,
        )
        self.assertFalse(dense_config.sparse_dynamics)
        self.assertFalse(dense_config.sparse_jacobian)
        self.assertEqual(dense_config.integrator, "euler")
        self.assertEqual(dense_config.dynamics.linear_solver_type, "LLTBRCM")
        self.assertEqual(dense_config.dvi.max_alternating_iterations, 20)

        padmm_config = SolverKamino.Config()
        self.assertFalse(padmm_config.sparse_dynamics)
        self.assertFalse(padmm_config.sparse_jacobian)
        self.assertEqual(padmm_config.integrator, "euler")

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(max_alternating_iterations=32, tolerance=1e-4),
        )
        self.assertEqual(config.dynamics_solver, "dvi")
        self.assertEqual(config.dvi.max_alternating_iterations, 32)
        self.assertEqual(config.dvi.inequality_sweeps_per_iteration, 1)
        self.assertEqual(config.dvi.bilateral_solve_interval, 1)
        self.assertEqual(config.dvi.contact_warmstart_method, "key_and_position_with_net_force_backup")
        self.assertFalse(config.dynamics.preconditioning)

        sparse_config = SolverKamino.Config(dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True)
        self.assertTrue(sparse_config.sparse_dynamics)
        self.assertTrue(sparse_config.sparse_jacobian)
        self.assertEqual(sparse_config.dvi.omega, 1.0)
        self.assertEqual(sparse_config.dynamics.linear_solver_type, "CR")
        self.assertEqual(sparse_config.dynamics.linear_solver_kwargs, {"maxiter": 9})
        with self.assertRaises(ValueError):
            SolverKamino.Config(
                dynamics_solver="dvi",
                dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=True),
            )
        invalid_dvi_configs = (
            {"tolerance": -1.0},
            {"regularization": 0.0},
            {"omega": 0.0},
            {"omega": 2.1},
            {"max_alternating_iterations": 0},
            {"inequality_sweeps_per_iteration": 0},
            {"bilateral_solve_interval": 0},
            {"bilateral_solver_type": "invalid"},
            {"warmstart_mode": "invalid"},
        )
        for kwargs in invalid_dvi_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                kamino_config.DVISolverConfig(**kwargs)
        for method in (
            "key_and_position",
            "geom_pair_net_force",
            "key_and_position_with_net_force_backup",
        ):
            self.assertEqual(
                kamino_config.DVISolverConfig(contact_warmstart_method=method).contact_warmstart_method, method
            )
        for method in ("reaction", "geom_pair_net_wrench"):
            with self.assertRaises(ValueError):
                kamino_config.DVISolverConfig(contact_warmstart_method=method)

        model_with_attrs = SimpleNamespace(
            kamino=SimpleNamespace(max_solver_iterations=wp.array([37], dtype=wp.int32, device=self.device))
        )
        self.assertEqual(kamino_config.DVISolverConfig.from_model(model_with_attrs).max_alternating_iterations, 37)

    def test_00a_dvi_contact_capacity_uses_geometry_heuristic(self):
        """Limit DVI contact allocation while honoring explicit overrides."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for box_index in range(16):
            body = builder.add_body(xform=wp.transform(wp.vec3(float(box_index), 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device=self.device)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            sparse_jacobian=False,
            use_collision_detector=True,
        )
        solver = SolverKamino(model, config)
        self.assertEqual(solver._contacts_kamino.world_max_contacts_host, [1000])
        self.assertLess(
            solver._contacts_kamino.model_max_contacts_host,
            solver._model_kamino.geoms.model_minimum_contacts,
        )
        override_config = SolverKamino.Config(
            dynamics_solver="dvi",
            sparse_jacobian=False,
            use_collision_detector=True,
            collision_detector=kamino_config.CollisionDetectorConfig(max_contacts_per_world=37),
        )
        override_solver = SolverKamino(model, override_config)
        self.assertEqual(override_solver._contacts_kamino.world_max_contacts_host, [37])

    def test_00a2_dvi_contact_capacity_reports_per_world_overflow(self):
        """Report contacts dropped when one world exhausts its capacity."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for world_index in range(50):
            builder.begin_world()
            if world_index == 0:
                for box_index in range(10):
                    body = builder.add_body(xform=wp.transform(wp.vec3(float(box_index), 0.0, 0.5), wp.quat_identity()))
                    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
            else:
                builder.add_body(is_kinematic=True)
            builder.end_world()
        model = builder.finalize(device=self.device)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            sparse_jacobian=False,
            use_collision_detector=True,
            collision_detector=kamino_config.CollisionDetectorConfig(max_contacts_per_world=1),
        )
        solver = SolverKamino(model, config)
        state_0 = model.state()
        state_1 = model.state()
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-3)
        self.assertGreater(
            int(solver._collision_detector_kamino._unified_pipeline.dropped_contact_count.numpy()[0]),
            0,
        )
        self.assertEqual(
            int(solver._collision_detector_kamino._unified_pipeline.contact_overflow_warning_emitted.numpy()[0]),
            1,
        )

    def test_00b_bilateral_solver_selection(self):
        """Verify DVI constructs and validates the configured bilateral solver."""

        def make_model(dimensions):
            return SimpleNamespace(
                size=SimpleNamespace(sum_of_num_joint_cts=sum(dimensions)),
                info=SimpleNamespace(
                    num_joint_cts=wp.array(dimensions, dtype=wp.int32, device=self.device),
                    joint_cts_offset=wp.array(np.cumsum([0, *dimensions[:-1]]), dtype=wp.int32, device=self.device),
                ),
            )

        config = kamino_config.DVISolverConfig(
            bilateral_solver_type="LLTBRCM",
            bilateral_solver_kwargs={"block_size": 16, "reuse_permutation": True},
        )
        solver = DVISolver()
        solver._config = [config]
        solver._data = SimpleNamespace(bilateral_operator=None, state=SimpleNamespace())
        solver._device = self.device
        solver._allocate_bilateral_solver(make_model([3]))

        self.assertIsInstance(solver._bilateral_solver, LLTBlockedRCMSolver)
        self.assertEqual(solver._bilateral_solver._block_size, 16)
        self.assertTrue(solver._bilateral_solver._reuse_permutation)

        solver._config = [
            kamino_config.DVISolverConfig(bilateral_solver_type="LLTB"),
            kamino_config.DVISolverConfig(bilateral_solver_type="LLTBRCM"),
        ]
        with self.assertRaisesRegex(ValueError, "All worlds must use the same"):
            solver._allocate_bilateral_solver(make_model([3, 3]))

    def test_00a_multiworld_status_reduction_requires_all_worlds_converged(self):
        """Require every world to converge when reducing DVI status."""
        status = np.array(
            [(True, 2, 1.0e-5), (False, 7, 2.0e-3)],
            dtype=[("converged", np.bool_), ("iterations", np.int32), ("r_d", np.float32)],
        )

        reduced = _reduce_solver_status(status)

        self.assertFalse(reduced["converged"])
        self.assertEqual(reduced["iterations"], 7)
        self.assertAlmostEqual(reduced["r_d"], 2.0e-3)

    def test_01_dvi_solve_dense_dual_problem(self):
        builder = basics.build_boxes_fourbar()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=0,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=None,
            jacobians=jacobians,
        )

        dynamics_config = kamino_config.ConstrainedDynamicsConfig(preconditioning=True)
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=DualProblem.Config(dynamics=dynamics_config),
            solver=LLTBlockedSolver,
            sparse=False,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)

        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                max_alternating_iterations=1000,
                inequality_sweeps_per_iteration=1,
                tolerance=1e-5,
                omega=1.0,
            ),
            warmstart=WarmStartMode.NONE,
            collect_info=True,
        )
        solver.reset()
        scratch = solver.data.state
        for array in (
            scratch.v_aug,
            scratch.s,
            scratch.scratch,
            scratch.bilateral_rhs,
            scratch.bilateral_solution,
            scratch.bilateral_preconditioner,
        ):
            array.fill_(float("nan"))
        scratch.bilateral_active_dim.fill_(-1)
        scratch.inequality_colors.fill_(-1)
        scratch.inequality_num_colors.fill_(-1)
        solver.coldstart()
        solver.solve(problem)
        _check_solution_matches_dual_problem(self, problem, solver)
        np.testing.assert_array_equal(solver.data.info.status.numpy(), solver.data.status.numpy())

    def test_01a_dvi_requires_contact_topology_at_allocation(self):
        """Reject missing contact topology when the model allocates contacts.

        Graph-colored inequality solves consume the contact container, so the
        omission must surface at allocation rather than inside a solve that may
        already be recorded into a captured graph.
        """
        model, data, _state, limits, detector, jacobians = make_containers(
            builder=basics.build_box_on_plane(),
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        self.assertGreater(model.size.max_of_max_contacts, 0)

        with self.assertRaises(ValueError):
            DVISolver(model=model, data=data, limits=limits, jacobians=jacobians)

        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
        )
        self.assertIsNotNone(solver.data)

    def test_02_public_solver_step_with_dvi(self):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.begin_world()
        body = builder.add_link(
            label="link",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_shape_box(label="box", body=body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            label="hinge",
            parent=-1,
            child=body,
            axis=newton.Axis.Y,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.device)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(
                max_alternating_iterations=500,
                inequality_sweeps_per_iteration=1,
                tolerance=1e-5,
            ),
            collect_solver_info=True,
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
        body_q = state_out.body_q.numpy()
        body_qd = state_out.body_qd.numpy()
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(body_qd)))
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)
        self.assertFalse(solver._solver_kamino.config.dynamics.preconditioning)
        self.assertIsNotNone(solver._solver_kamino.solver_fd.data.info)

    def test_02a_public_solver_reset_accepts_global_mask_slot(self):
        """Accept a global reset slot as a no-op for Kamino state."""
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.device)
        solver = SolverKamino(model)
        state = model.state()

        body_q = state.body_q.numpy()
        body_q[:, 0] += 1.0
        state.body_q.assign(body_q)
        solver.reset(
            state,
            world_mask=wp.array((False, True), dtype=wp.bool, device=self.device),
            flags=0,
        )

        np.testing.assert_array_equal(state.body_q.numpy(), body_q)

    def test_03_dvi_solve_single_contact(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=1,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.model_active_contacts.numpy()[0]), 0)

        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            solver=LLTBlockedSolver,
            sparse=False,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)

        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                max_alternating_iterations=200,
                inequality_sweeps_per_iteration=1,
                tolerance=1e-4,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertLessEqual(int(status["iterations"]), _status_iteration_budget(solver, 0))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))

    def test_03l_sparse_dvi_skips_inequalities_without_mapped_topology(self):
        """Skip a sparse inequality whose row has no mapped Jacobian topology.

        The sweep reads per-entity Jacobian offsets through the mapped index,
        so an unmapped row must be skipped rather than dereferenced through a
        negative index.
        """

        def solve_single_limit(limit_index: int) -> float:
            int32_array = lambda values: wp.array(values, dtype=wp.int32, device=self.device)  # noqa: E731
            float_array = lambda values: wp.array(values, dtype=wp.float32, device=self.device)  # noqa: E731
            jacobian_block = wp.array([vec6f(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=vec6f, device=self.device)
            lambdas = float_array([0.0])
            config = wp.array(
                [
                    convert_config_to_struct(
                        kamino_config.DVISolverConfig(
                            max_alternating_iterations=1,
                            inequality_sweeps_per_iteration=1,
                            tolerance=0.0,
                            regularization=1e-6,
                        )
                    )
                ],
                dtype=DVIConfigStruct,
                device=self.device,
            )
            threads_per_world = 64 if self.device.is_cuda else 1
            wp.launch(
                kernel=_solve_dvi_sparse_inequalities_pgs,
                dim=threads_per_world,
                inputs=[
                    int32_array([1]),  # bsm_num_nzb
                    int32_array([0]),  # bsm_nzb_start
                    wp.array([[0, 0]], dtype=wp.int32, device=self.device),  # bsm_nzb_coords
                    jacobian_block,  # bsm_nzb_values
                    jacobian_block,  # jacobian_nzb_values
                    int32_array([0]),  # bsm_row_start
                    int32_array([0]),  # bsm_col_start
                    int32_array([0]),  # limit_nzb_offsets
                    int32_array([0]),  # contact_nzb_offsets
                    int32_array([limit_index]),  # limit_indices
                    int32_array([-1]),  # contact_indices
                    int32_array([1]),  # problem_nl
                    int32_array([0]),  # problem_nc
                    int32_array([0]),  # problem_lio
                    int32_array([0]),  # problem_cio
                    int32_array([0]),  # problem_uio
                    int32_array([0]),  # problem_lcgo
                    int32_array([1]),  # problem_ccgo
                    int32_array([0]),  # problem_vio
                    float_array([0.0]),  # problem_mu
                    float_array([1.0]),  # problem_P
                    float_array([-1.0]),  # problem_v_f
                    float_array([1.0]),  # problem_diag
                    float_array([0.0]),  # eta
                    int32_array([1]),  # inequality_num_colors
                    int32_array([0]),  # inequality_ids_by_color
                    int32_array([0, 1]),  # inequality_color_starts
                    -1,  # block_iteration
                    config,
                    wp.zeros(6, dtype=wp.float32, device=self.device),  # body_space
                    lambdas,
                ],
                device=self.device,
                block_dim=threads_per_world,
            )
            return float(lambdas.numpy()[0])

        # A mapped row resolves its violated limit velocity into a positive impulse.
        self.assertAlmostEqual(solve_single_limit(0), 1.0, places=4)
        # An unmapped row keeps its impulse and touches no Jacobian offsets.
        self.assertEqual(solve_single_limit(-1), 0.0)

    def _make_box_on_plane_setup(self, max_world_contacts: int = 4):
        """Build an inequality-only box-on-plane problem and its containers."""
        model, data, state, limits, detector, jacobians = make_containers(
            builder=basics.build_box_on_plane(),
            device=self.device,
            max_world_contacts=max_world_contacts,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        setup = SimpleNamespace(data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)
        return model, problem, setup

    def test_03j_dvi_omega_scales_projected_updates_without_moving_the_solution(self):
        """Relax projected updates by `omega` while preserving the fixed point.

        `omega` scales the step of every projected update, so a single sweep
        must move less for a smaller value, while additional sweeps must still
        reach the same cone-complementarity solution.
        """
        model, problem, setup = self._make_box_on_plane_setup()

        def solve_normal_impulse(omega: float, sweep_count: int) -> float:
            solver = _solve_dvi(
                model,
                problem,
                config=kamino_config.DVISolverConfig(
                    max_alternating_iterations=sweep_count,
                    inequality_sweeps_per_iteration=1,
                    tolerance=0.0,
                    regularization=1e-6,
                    omega=omega,
                ),
                setup=setup,
            )
            lambdas = solver.data.solution.lambdas.numpy()
            offset = int(problem.data.vio.numpy()[0] + problem.data.ccgo.numpy()[0])
            count = int(problem.data.nc.numpy()[0])
            return float(np.sum(lambdas[offset + 2 : offset + 3 * count : 3]))

        single_sweep_slow = solve_normal_impulse(0.25, sweep_count=1)
        single_sweep_fast = solve_normal_impulse(1.0, sweep_count=1)
        self.assertGreater(single_sweep_slow, 0.0)
        self.assertGreater(single_sweep_fast, 2.0 * single_sweep_slow)

        converged_slow = solve_normal_impulse(0.25, sweep_count=400)
        converged_fast = solve_normal_impulse(1.0, sweep_count=400)
        self.assertAlmostEqual(converged_slow, converged_fast, delta=1.0e-4 * converged_fast)

    def test_03k_dvi_inequality_only_status_reports_the_sweep_budget(self):
        """Report the sweeps actually run by the inequality-only dense path."""
        model, problem, setup = self._make_box_on_plane_setup()
        max_alternating_iterations = 17
        inequality_sweeps_per_iteration = 3
        solver = _solve_dvi(
            model,
            problem,
            config=kamino_config.DVISolverConfig(
                max_alternating_iterations=max_alternating_iterations,
                tolerance=1e-4,
                regularization=1e-6,
                inequality_sweeps_per_iteration=inequality_sweeps_per_iteration,
            ),
            setup=setup,
        )

        self.assertTrue(solver._can_use_dense_inequality_pgs())
        self.assertEqual(
            int(solver.data.status.numpy()[0]["iterations"]),
            max_alternating_iterations * inequality_sweeps_per_iteration,
        )

    def test_03d_dvi_direct_block_honors_per_world_iteration_counts(self):
        """Honor each world's projected and bilateral iteration schedule."""
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=3,
            build_fn=basics.build_boxes_hinged,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertTrue(np.all(detector.contacts.world_active_contacts.numpy() > 0))

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        configs = [
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                max_alternating_iterations=1,
                inequality_sweeps_per_iteration=1,
            ),
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                max_alternating_iterations=3,
                inequality_sweeps_per_iteration=1,
                bilateral_solve_interval=2,
            ),
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                max_alternating_iterations=1,
                inequality_sweeps_per_iteration=3,
            ),
        ]

        def solve_normal_sums() -> list[float]:
            solver = DVISolver(
                model=model,
                data=data,
                limits=limits,
                contacts=detector.contacts,
                jacobians=jacobians,
                config=configs,
                warmstart=WarmStartMode.NONE,
            )
            self.assertEqual(solver._bilateral_solve_after_block, (False, True))
            zero_dims = np.zeros(3, dtype=np.int32)
            joint_dims = problem.data.njc.numpy()
            solver._set_bilateral_active_dim(problem, 0)
            np.testing.assert_array_equal(solver.data.state.bilateral_active_dim.numpy(), zero_dims)
            solver._set_bilateral_active_dim(problem, 1)
            np.testing.assert_array_equal(
                solver.data.state.bilateral_active_dim.numpy(),
                np.array([0, joint_dims[1], 0], dtype=np.int32),
            )
            solver._set_bilateral_active_dim(problem, -1)
            np.testing.assert_array_equal(solver.data.state.bilateral_active_dim.numpy(), joint_dims)
            active_dim_updates = []
            set_bilateral_active_dim = solver._set_bilateral_active_dim

            def record_bilateral_active_dim(problem: DualProblem, block_iteration: int) -> None:
                set_bilateral_active_dim(problem, block_iteration)
                active_dim_updates.append((block_iteration, solver.data.state.bilateral_active_dim.numpy().copy()))

            solver._set_bilateral_active_dim = record_bilateral_active_dim
            solver.reset()
            solver.coldstart()
            solver.solve(problem)
            self.assertEqual([block_iteration for block_iteration, _ in active_dim_updates], [1, -1])
            np.testing.assert_array_equal(
                active_dim_updates[0][1],
                np.array([0, joint_dims[1], 0], dtype=np.int32),
            )
            np.testing.assert_array_equal(active_dim_updates[1][1], joint_dims)
            status = solver.data.status.numpy()
            self.assertEqual([int(status[wid]["iterations"]) for wid in range(3)], [1, 3, 3])
            self.assertTrue(np.all(solver.data.state.inequality_num_colors.numpy() > 0))
            np.testing.assert_array_equal(
                solver.data.state.bilateral_active_dim.numpy(),
                problem.data.njc.numpy(),
            )

            lambdas = extract_problem_vector(
                problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
            )
            ccgo = problem.data.ccgo.numpy().astype(int)
            nc = problem.data.nc.numpy().astype(int)
            return [float(np.sum(lambdas[wid][ccgo[wid] + 2 : ccgo[wid] + 3 * nc[wid] : 3])) for wid in range(3)]

        normal_sums = solve_normal_sums()
        self.assertGreater(normal_sums[1], normal_sums[0])
        self.assertGreater(normal_sums[2], normal_sums[0])

    def test_03d1_sparse_dvi_honors_per_world_bilateral_intervals(self):
        """Restrict sparse bilateral re-solves to each world's configured interval."""
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=2,
            build_fn=basics.build_boxes_hinged,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
            sparse=True,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        problem = _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
        configs = [
            kamino_config.DVISolverConfig(
                max_alternating_iterations=3,
                bilateral_solve_interval=1,
            ),
            kamino_config.DVISolverConfig(
                max_alternating_iterations=3,
                bilateral_solve_interval=99,
            ),
        ]
        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            problem=problem,
            config=configs,
            warmstart=WarmStartMode.NONE,
        )
        active_dim_updates = []
        set_bilateral_active_dim = solver._sparse_path.set_bilateral_active_dim

        def record_bilateral_active_dim(problem: DualProblem, block_iteration: int) -> None:
            set_bilateral_active_dim(problem, block_iteration)
            active_dim_updates.append((block_iteration, solver.data.state.bilateral_active_dim.numpy().copy()))

        solver._sparse_path.set_bilateral_active_dim = record_bilateral_active_dim
        solver.coldstart()
        solver.solve(problem)

        joint_dims = problem.data.njc.numpy()
        self.assertEqual([block_iteration for block_iteration, _ in active_dim_updates], [0, 1, -1])
        np.testing.assert_array_equal(
            active_dim_updates[0][1],
            np.array([joint_dims[0], 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(active_dim_updates[1][1], active_dim_updates[0][1])
        np.testing.assert_array_equal(active_dim_updates[2][1], joint_dims)

    def test_03d2_dvi_direct_block_finishes_with_bilateral_solve(self):
        builder = basics.build_boxes_hinged()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.world_active_contacts.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                max_alternating_iterations=1,
                inequality_sweeps_per_iteration=1,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        v_plus = extract_problem_vector(problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=True)[0]
        njc = int(problem.data.njc.numpy()[0])
        status = solver.data.status.numpy()[0]

        self.assertGreater(njc, 0)
        self.assertLess(float(np.max(np.abs(v_plus[:njc]))), 1e-6)
        self.assertLess(float(status["r_b"]), 1e-6)

    def test_03e_dvi_direct_block_no_unilateral_rows_reports_single_iteration(self):
        builder = basics.build_box_pendulum(ground=False)
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        self.assertGreater(int(model.info.num_joint_cts.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nl.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nc.numpy()[0]), 0)

        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                tolerance=1e-4,
                regularization=1e-5,
                max_alternating_iterations=7,
                inequality_sweeps_per_iteration=3,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertEqual(int(status["iterations"]), 1)
        self.assertEqual(int(solver.data.state.bilateral_active_dim.numpy()[0]), 0)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_03f_dvi_bilateral_only_solve_resets_stale_status(self):
        builder = basics.build_box_pendulum(ground=False)
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=0,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=None,
            jacobians=jacobians,
        )

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        self.assertGreater(int(model.info.num_joint_cts.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nl.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nc.numpy()[0]), 0)

        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                tolerance=1e-4,
                regularization=1e-5,
                inequality_sweeps_per_iteration=5,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        wp.launch(
            kernel=_initialize_dvi_status,
            dim=solver.size.num_worlds,
            inputs=[
                solver.data.config,
                solver.data.status,
            ],
            device=self.device,
        )
        self.assertEqual(int(solver.data.status.numpy()[0]["iterations"]), 5)

        solver.solve(problem)
        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertEqual(int(status["iterations"]), 1)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_03g_dvi_inequality_coloring_separates_dynamic_conflicts(self):
        """Separate conflicting inequality endpoints while sharing safe colors."""
        problem_nl = wp.array([1], dtype=wp.int32, device=self.device)
        problem_nc = wp.array([4], dtype=wp.int32, device=self.device)
        problem_uio = wp.array([0], dtype=wp.int32, device=self.device)
        contact_bid_ab = wp.array(
            [
                wp.vec2i(0, -1),
                wp.vec2i(0, 1),
                wp.vec2i(2, -1),
                wp.vec2i(-1, -1),
                wp.vec2i(1, -1),
            ],
            dtype=wp.vec2i,
            device=self.device,
        )
        body_color_masks = wp.zeros(shape=3, dtype=wp.uint64, device=self.device)
        inequality_colors = wp.full(shape=5, value=-1, dtype=wp.int32, device=self.device)
        inequality_num_colors = wp.zeros(shape=1, dtype=wp.int32, device=self.device)
        inequality_ids_by_color = wp.full(shape=5, value=-1, dtype=wp.int32, device=self.device)
        inequality_color_starts = wp.zeros(shape=6, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=_color_mapped_dvi_inequalities,
            dim=1,
            inputs=[
                problem_nl,
                problem_nc,
                problem_uio,
                contact_bid_ab,
                body_color_masks,
                inequality_colors,
                inequality_num_colors,
                inequality_ids_by_color,
                inequality_color_starts,
            ],
            device=self.device,
        )
        colors = inequality_colors.numpy()
        num_colors = int(inequality_num_colors.numpy()[0])
        self.assertGreaterEqual(num_colors, 2)
        self.assertTrue(np.all(colors >= 0))
        self.assertNotEqual(colors[0], colors[1])
        self.assertNotEqual(colors[1], colors[4])
        self.assertLess(colors[3], num_colors)
        ids_by_color = inequality_ids_by_color.numpy()
        color_starts = inequality_color_starts.numpy()
        np.testing.assert_array_equal(np.sort(ids_by_color), np.arange(5))
        for color in range(num_colors):
            scheduled = ids_by_color[color_starts[color] : color_starts[color + 1]]
            self.assertTrue(np.all(colors[scheduled] == color))

    def test_03g1_dvi_inequality_coloring_keeps_worlds_independent(self):
        """Color independent worlds concurrently without sharing body masks."""
        problem_nl = wp.array([0, 0], dtype=wp.int32, device=self.device)
        problem_nc = wp.array([2, 2], dtype=wp.int32, device=self.device)
        problem_uio = wp.array([0, 2], dtype=wp.int32, device=self.device)
        inequality_bodies = wp.array(
            [wp.vec2i(0, -1), wp.vec2i(0, -1), wp.vec2i(1, -1), wp.vec2i(1, -1)],
            dtype=wp.vec2i,
            device=self.device,
        )
        body_color_masks = wp.zeros(shape=2, dtype=wp.uint64, device=self.device)
        inequality_colors = wp.full(shape=4, value=-1, dtype=wp.int32, device=self.device)
        inequality_num_colors = wp.zeros(shape=2, dtype=wp.int32, device=self.device)
        inequality_ids_by_color = wp.full(shape=4, value=-1, dtype=wp.int32, device=self.device)
        inequality_color_starts = wp.zeros(shape=6, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=_color_mapped_dvi_inequalities,
            dim=2,
            inputs=[
                problem_nl,
                problem_nc,
                problem_uio,
                inequality_bodies,
                body_color_masks,
                inequality_colors,
                inequality_num_colors,
                inequality_ids_by_color,
                inequality_color_starts,
            ],
            device=self.device,
        )

        np.testing.assert_array_equal(inequality_colors.numpy(), [0, 1, 0, 1])
        np.testing.assert_array_equal(inequality_num_colors.numpy(), [2, 2])
        np.testing.assert_array_equal(inequality_ids_by_color.numpy(), [0, 1, 0, 1])
        np.testing.assert_array_equal(inequality_color_starts.numpy(), [0, 1, 2, 0, 1, 2])

    def test_03g2_dvi_inequality_coloring_handles_more_than_64_colors(self):
        """Preserve valid coloring when one body requires more than 64 colors."""
        num_inequalities = 66
        problem_nl = wp.array([0], dtype=wp.int32, device=self.device)
        problem_nc = wp.array([num_inequalities], dtype=wp.int32, device=self.device)
        problem_uio = wp.array([0], dtype=wp.int32, device=self.device)
        inequality_bodies = wp.array(
            [wp.vec2i(0, -1)] * num_inequalities,
            dtype=wp.vec2i,
            device=self.device,
        )
        body_color_masks = wp.zeros(shape=1, dtype=wp.uint64, device=self.device)
        inequality_colors = wp.full(
            shape=num_inequalities,
            value=-1,
            dtype=wp.int32,
            device=self.device,
        )
        inequality_num_colors = wp.zeros(shape=1, dtype=wp.int32, device=self.device)
        inequality_ids_by_color = wp.full(shape=num_inequalities, value=-1, dtype=wp.int32, device=self.device)
        inequality_color_starts = wp.zeros(shape=num_inequalities + 1, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=_color_mapped_dvi_inequalities,
            dim=1,
            inputs=[
                problem_nl,
                problem_nc,
                problem_uio,
                inequality_bodies,
                body_color_masks,
                inequality_colors,
                inequality_num_colors,
                inequality_ids_by_color,
                inequality_color_starts,
            ],
            device=self.device,
        )

        np.testing.assert_array_equal(inequality_colors.numpy(), np.arange(num_inequalities))
        self.assertEqual(int(inequality_num_colors.numpy()[0]), num_inequalities)

        np.testing.assert_array_equal(inequality_ids_by_color.numpy(), np.arange(num_inequalities))
        np.testing.assert_array_equal(inequality_color_starts.numpy(), np.arange(num_inequalities + 1))

    def test_03i_dvi_coldstart_is_repeatable(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(
                    builder_fn=basics.build_boxes_hinged,
                    max_world_contacts=8,
                    gravity=True,
                    perturb=True,
                    device=self.device,
                    sparse=sparse,
                )
                test.build()
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    sparse_dynamics=sparse,
                    sparse_jacobian=sparse,
                ).dvi
                solver = _solve_dvi(test.model, test.problem, config=config, setup=test)
                first_lambdas = solver.data.solution.lambdas.numpy().copy()
                first_v_plus = solver.data.solution.v_plus.numpy().copy()
                first_status = solver.data.status.numpy().copy()

                test.build()
                solver.reset()
                solver.coldstart()
                solver.solve(test.problem)

                np.testing.assert_allclose(solver.data.solution.lambdas.numpy(), first_lambdas, rtol=0.0, atol=1e-6)
                np.testing.assert_allclose(solver.data.solution.v_plus.numpy(), first_v_plus, rtol=0.0, atol=1e-6)
                status = solver.data.status.numpy()
                np.testing.assert_array_equal(status["converged"], first_status["converged"])
                np.testing.assert_array_equal(status["iterations"], first_status["iterations"])
                for residual in ("r_p", "r_d", "r_c", "r_b"):
                    np.testing.assert_allclose(status[residual], first_status[residual], rtol=1e-5, atol=1e-8)

    def test_04_dvi_solve_active_joint_limit(self):
        """Resolve an active joint limit through the inequality solver."""
        builder = testing.build_unary_revolute_joint_test(limits=True, ground=False)
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=0,
            sparse=False,
        )
        update_containers(model=model, data=data, state=state, limits=limits, detector=None, jacobians=None)

        q_j = data.joints.q_j.numpy()
        q_j[:] = 1.0
        data.joints.q_j.assign(q_j)
        limits.detect(q_j=data.joints.q_j)
        update_constraints_info(model=model, data=data)
        jacobians.build(model=model, data=data, limits=limits.data, contacts=None)
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(
            model,
            problem,
            config=kamino_config.DVISolverConfig(
                max_alternating_iterations=32,
                inequality_sweeps_per_iteration=4,
                tolerance=1e-4,
                regularization=1e-5,
            ),
            setup=SimpleNamespace(data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians),
        )

        _assert_solver_status_converged(self, solver)
        iterations = int(solver.data.status.numpy()[0]["iterations"])
        self.assertEqual(
            iterations, solver.config[0].max_alternating_iterations * solver.config[0].inequality_sweeps_per_iteration
        )
        self.assertEqual(iterations, 128)
        _check_solution_matches_dual_problem(self, problem, solver)
        limit_offset = int(problem.data.lcgo.numpy()[0])
        limit_impulse = float(solver.data.solution.lambdas.numpy()[limit_offset])
        limit_velocity = float(solver.data.solution.v_plus.numpy()[limit_offset])
        self.assertGreater(limit_impulse, 1.0e-4)
        self.assertGreaterEqual(limit_velocity, -solver.config[0].tolerance)
        self.assertLessEqual(abs(limit_impulse * limit_velocity), solver.config[0].tolerance)

    def test_07_dvi_singular_limit_rows_remain_finite(self):
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.device,
            max_world_contacts=0,
            with_limits=True,
            with_contacts=False,
        )
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians.build(model=model, data=data, limits=limits.data, contacts=None)
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, contacts, jacobians)
        solver = _solve_dvi(
            model,
            problem,
            setup=SimpleNamespace(data=data, limits=limits, contacts=contacts, jacobians=jacobians),
        )

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 0)
        _assert_solution_finite(self, solver)
        lambdas_np = extract_problem_vector(
            problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
        )[0]
        limit_start = int(problem.data.lcgo.numpy()[0])
        limit_count = int(problem.data.nl.numpy()[0])
        limit_lambdas = lambdas_np[limit_start : limit_start + limit_count]
        self.assertLess(float(np.max(np.abs(limit_lambdas))), 1.0)

    def test_08_public_solver_short_rollout_with_dvi(self):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.begin_world()
        body = builder.add_link(
            label="link",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_shape_box(label="box", body=body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            label="hinge",
            parent=-1,
            child=body,
            axis=newton.Axis.Y,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.device)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(
                max_alternating_iterations=300,
                inequality_sweeps_per_iteration=1,
                tolerance=1e-4,
            ),
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        for _ in range(8):
            solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
            state_in, state_out = state_out, state_in
        self.assertTrue(np.all(np.isfinite(state_in.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_in.body_qd.numpy())))
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)

    def test_08a_public_solver_heterogeneous_contact_rollout_with_dvi(self):
        """Use Cholesky for heterogeneous dense and sparse DVI rollouts."""
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        public_basics.make_basics_heterogeneous_builder(builder=builder, ground=True)
        model = builder.finalize(device=self.device, skip_validation_joints=True)

        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    sparse_dynamics=sparse,
                    sparse_jacobian=True,
                    use_collision_detector=True,
                    collision_detector=kamino_config.CollisionDetectorConfig(
                        max_contacts=64 * model.world_count,
                        max_contacts_per_world=64,
                        max_contacts_per_pair=16,
                    ),
                )
                solver = SolverKamino(model, config=config)
                state_in = model.state()
                state_out = model.state()
                control = model.control()

                contact_seen = False
                for _ in range(24):
                    solver.step(state_in, state_out, control=control, contacts=None, dt=1.0e-3)
                    state_in, state_out = state_out, state_in
                    contact_seen = contact_seen or bool(
                        np.any(solver._contacts_kamino.world_active_contacts.numpy() > 0)
                    )

                dvi_solver = solver._solver_kamino.solver_fd
                status = dvi_solver.data.status.numpy()
                self.assertTrue(contact_seen)
                self.assertTrue(np.all(np.isfinite(state_in.body_q.numpy())))
                self.assertTrue(np.all(np.isfinite(state_in.body_qd.numpy())))
                self.assertTrue(np.all(np.isfinite(status["r_p"])))
                self.assertTrue(np.all(np.isfinite(status["r_d"])))
                self.assertLess(float(np.max(np.abs(state_in.body_qd.numpy()))), 100.0)
                self.assertIsInstance(dvi_solver, DVISolver)
                self.assertIsInstance(dvi_solver._bilateral_solver, (LLTBlockedSolver, LLTBlockedRCMSolver))
                joint_dims = solver._solver_kamino._model.info.num_joint_cts.numpy()
                self.assertTrue(np.any(joint_dims == 0))
                self.assertTrue(np.any(joint_dims > 0))
                np.testing.assert_array_equal(
                    dvi_solver.data.bilateral_operator.info.dimensions,
                    np.maximum(joint_dims, 1),
                )
                self.assertEqual(config.sparse_dynamics, sparse)
                self.assertTrue(config.sparse_jacobian)

    def test_03a_sparse_dvi_filtered_matvec_matches_full_rows(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=True,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.model_active_contacts.numpy()[0]), 0)

        problem = _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                max_alternating_iterations=1,
                inequality_sweeps_per_iteration=1,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()

        lambdas = np.linspace(-0.25, 0.5, problem.data.v_f.shape[0], dtype=np.float32)
        solver.data.solution.lambdas.assign(lambdas)

        full = wp.zeros_like(problem.data.v_f)
        problem.delassus.matvec(solver.data.solution.lambdas, full, solver.all_worlds_mask)
        full_np = full.numpy()

        _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_JOINTS)
        joint_np = solver.data.state.v_aug.numpy()
        _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_UNILATERAL)
        unilateral_np = solver.data.state.v_aug.numpy()

        dim = int(problem.data.dim.numpy()[0])
        njc = int(problem.data.njc.numpy()[0])
        np.testing.assert_allclose(joint_np[:njc], full_np[:njc], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(unilateral_np[njc:dim], full_np[njc:dim], rtol=1e-5, atol=1e-5)

    def test_05_dvi_solve_multi_world_contacts(self):
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=4,
            build_fn=basics.build_box_on_plane,
            ground=True,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertTrue(np.all(detector.contacts.world_active_contacts.numpy() > 0))

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(
            model,
            problem,
            setup=SimpleNamespace(data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians),
        )

        _assert_solver_status_converged(self, solver)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_05a_dvi_maps_packed_multiworld_contacts(self):
        """Verify dense and sparse DVI map packed contacts to raw topology."""
        for sparse in (False, True):
            builder = builder_utils.make_homogeneous_builder(
                num_worlds=8,
                build_fn=basics.build_box_on_plane,
                ground=True,
            )
            model, data, state, limits, detector, jacobians = make_containers(
                builder=builder,
                device=self.device,
                max_world_contacts=4,
                sparse=sparse,
            )
            update_containers(
                model=model,
                data=data,
                state=state,
                limits=limits,
                detector=detector,
                jacobians=jacobians,
            )
            problem = (
                _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
                if sparse
                else _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
            )
            solver = DVISolver(
                model=model,
                data=data,
                limits=limits,
                contacts=detector.contacts,
                jacobians=jacobians,
                config=kamino_config.DVISolverConfig(
                    max_alternating_iterations=100,
                    inequality_sweeps_per_iteration=1,
                    tolerance=1e-5,
                ),
                warmstart=WarmStartMode.NONE,
            )
            solver.coldstart()
            solver.solve(problem)

            contact_indices = solver.data.state.contact_indices.numpy()
            contact_wid = detector.contacts.wid.numpy()
            contact_cid = detector.contacts.cid.numpy()
            problem_nc = problem.data.nc.numpy()
            problem_cio = problem.data.cio.numpy()
            for wid, nc in enumerate(problem_nc):
                for cid in range(int(nc)):
                    raw_contact = int(contact_indices[int(problem_cio[wid]) + cid])
                    self.assertGreaterEqual(raw_contact, 0)
                    self.assertEqual(int(contact_wid[raw_contact]), wid)
                    self.assertEqual(int(contact_cid[raw_contact]), cid)

            _assert_solver_status_converged(self, solver)

    def test_05b_dvi_five_box_stack_converges_within_budget(self):
        """Converge a coupled five-box stack in dense and sparse modes."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = builder_utils.make_homogeneous_builder(4, _build_five_box_stack)
                model, data, state, limits, detector, jacobians = make_containers(
                    builder=builder,
                    device=self.device,
                    max_world_contacts=64,
                    sparse=sparse,
                    dt=1.0e-3,
                )
                update_containers(model, data, state, limits, detector, jacobians)
                self.assertTrue(np.all(detector.contacts.world_active_contacts.numpy() == 36))

                problem = (
                    _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
                    if sparse
                    else _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
                )
                solver = _solve_dvi(
                    model,
                    problem,
                    config=kamino_config.DVISolverConfig(
                        max_alternating_iterations=75,
                        inequality_sweeps_per_iteration=1,
                        tolerance=1.0e-5,
                        regularization=1.0e-6,
                    ),
                    setup=SimpleNamespace(
                        data=data,
                        limits=limits,
                        contacts=detector.contacts,
                        jacobians=jacobians,
                    ),
                )

                _assert_solver_status_converged(self, solver)
                _check_solution_matches_dual_problem(self, problem, solver)

                contact_indices = solver.data.state.contact_indices.numpy()
                contact_bodies = detector.contacts.bid_AB.numpy()
                lambdas = solver.data.solution.lambdas.numpy()
                problem_nc = problem.data.nc.numpy()
                problem_cio = problem.data.cio.numpy()
                problem_ccgo = problem.data.ccgo.numpy()
                problem_vio = problem.data.vio.numpy()
                expected_ground_impulse = 5.0 * 9.81e-3
                expected_total_impulse = 15.0 * 9.81e-3
                for world, contact_count in enumerate(problem_nc):
                    count = int(contact_count)
                    contact_offset = int(problem_cio[world])
                    constraint_offset = int(problem_vio[world] + problem_ccgo[world])
                    raw_contacts = contact_indices[contact_offset : contact_offset + count]
                    bodies = contact_bodies[raw_contacts]
                    normal_impulses = lambdas[constraint_offset + 2 : constraint_offset + 3 * count : 3]
                    ground_contacts = np.any(bodies == -1, axis=1)
                    self.assertAlmostEqual(
                        float(np.sum(normal_impulses[ground_contacts])),
                        expected_ground_impulse,
                        delta=0.02 * expected_ground_impulse,
                    )
                    self.assertAlmostEqual(
                        float(np.sum(normal_impulses)),
                        expected_total_impulse,
                        delta=0.02 * expected_total_impulse,
                    )

    def test_05b2_dvi_reduced_kapla_tower_remains_stable(self):
        """Keep a six-layer plank tower stable in dense and sparse modes."""
        builder, bodies, initial_positions = _build_reduced_kapla_tower()
        model = builder.finalize(device=self.device)
        dt = 1.0e-3
        steps = 50
        final_positions = []

        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    use_collision_detector=True,
                    sparse_dynamics=sparse,
                    sparse_jacobian=sparse,
                    collision_detector=kamino_config.CollisionDetectorConfig(
                        max_contacts=512,
                        max_contacts_per_world=512,
                        max_contacts_per_pair=8,
                    ),
                )
                solver = SolverKamino(model, config=config)
                state_0 = model.state()
                state_1 = model.state()
                if self.device.is_cuda and wp.is_mempool_enabled(self.device):
                    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                    solver.step(state_1, state_0, control=None, contacts=None, dt=dt)
                    with wp.ScopedCapture(self.device) as capture:
                        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                        solver.step(state_1, state_0, control=None, contacts=None, dt=dt)
                    for _ in range((steps - 4) // 2):
                        wp.capture_launch(capture.graph)
                else:
                    for _ in range(steps):
                        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                        state_0, state_1 = state_1, state_0

                _assert_solver_status_converged(self, solver._solver_kamino.solver_fd)
                positions = state_0.body_q.numpy()[bodies, :3]
                velocities = state_0.body_qd.numpy()[bodies]
                drop = initial_positions[:, 2] - positions[:, 2]
                horizontal_drift = np.linalg.norm(positions[:, :2] - initial_positions[:, :2], axis=1)

                contact_count = int(solver._contacts_kamino.world_active_contacts.numpy()[0])
                self.assertGreater(contact_count, len(bodies))
                self.assertLess(contact_count, 512)
                self.assertTrue(np.all(np.isfinite(positions)))
                self.assertTrue(np.all(np.isfinite(velocities)))
                self.assertLess(float(np.max(drop)), 0.015)
                self.assertLess(float(np.max(horizontal_drift)), 0.01)
                final_positions.append(positions)

        if len(final_positions) == 2:
            np.testing.assert_allclose(final_positions[0], final_positions[1], rtol=0.0, atol=2.0e-3)

    def test_05c_dvi_high_mass_ratio_stack_supports_weight(self):
        """Support a 100:1 sphere stack accurately in dense and sparse modes."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, data, state, limits, detector, jacobians = make_containers(
                    builder=_build_high_mass_ratio_sphere_stack(),
                    device=self.device,
                    max_world_contacts=4,
                    sparse=sparse,
                    dt=1.0e-3,
                )
                update_containers(model, data, state, limits, detector, jacobians)
                self.assertEqual(int(detector.contacts.world_active_contacts.numpy()[0]), 2)

                problem = (
                    _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
                    if sparse
                    else _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
                )
                solver = _solve_dvi(
                    model,
                    problem,
                    config=kamino_config.DVISolverConfig(
                        max_alternating_iterations=500,
                        inequality_sweeps_per_iteration=1,
                        tolerance=1.0e-4,
                        regularization=1.0e-6,
                    ),
                    setup=SimpleNamespace(
                        data=data,
                        limits=limits,
                        contacts=detector.contacts,
                        jacobians=jacobians,
                    ),
                )
                _assert_solver_status_converged(self, solver)

                contact_indices = solver.data.state.contact_indices.numpy()
                contact_bodies = detector.contacts.bid_AB.numpy()[contact_indices[:2]]
                normal_impulses = solver.data.solution.lambdas.numpy()[2:6:3]
                ground_contact = np.any(contact_bodies == -1, axis=1)
                expected_ground_impulse = 101.0 * 9.81e-3
                expected_pair_impulse = 100.0 * 9.81e-3
                self.assertAlmostEqual(
                    float(normal_impulses[ground_contact][0]),
                    expected_ground_impulse,
                    delta=0.02 * expected_ground_impulse,
                )
                self.assertAlmostEqual(
                    float(normal_impulses[~ground_contact][0]),
                    expected_pair_impulse,
                    delta=0.02 * expected_pair_impulse,
                )

    def test_05d_dvi_colors_contacts_with_joint_limits(self):
        """Solve contacts and joint limits through one colored inequality path."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = _build_five_box_stack()
                testing.build_unary_revolute_joint_test(
                    builder=builder,
                    z_offset=5.0,
                    new_world=False,
                    limits=True,
                    ground=False,
                    world_index=0,
                )
                model, data, state, limits, detector, jacobians = make_containers(
                    builder=builder,
                    device=self.device,
                    max_world_contacts=64,
                    sparse=sparse,
                    dt=1.0e-3,
                )
                update_containers(
                    model=model,
                    data=data,
                    state=state,
                    limits=limits,
                    detector=detector,
                    jacobians=jacobians,
                )
                joint_q = data.joints.q_j.numpy()
                joint_q[0] = 1.0
                data.joints.q_j.assign(joint_q)
                limits.detect(q_j=data.joints.q_j)
                update_constraints_info(model=model, data=data)
                jacobians.build(model=model, data=data, limits=limits.data, contacts=detector.contacts)
                if sparse:
                    problem = _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
                else:
                    problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
                solver = _solve_dvi(
                    model,
                    problem,
                    config=kamino_config.DVISolverConfig(
                        max_alternating_iterations=64,
                        inequality_sweeps_per_iteration=1,
                        tolerance=1.0e-5,
                        regularization=1.0e-6,
                    ),
                    setup=SimpleNamespace(data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians),
                )

                self.assertEqual(int(limits.model_active_limits.numpy()[0]), 1)
                self.assertEqual(int(detector.contacts.world_active_contacts.numpy()[0]), 36)
                self.assertGreater(int(solver.data.state.inequality_num_colors.numpy()[0]), 0)

                count = int(problem.data.nc.numpy()[0])
                raw_contacts = solver.data.state.contact_indices.numpy()[:count]
                contact_bodies = detector.contacts.bid_AB.numpy()[raw_contacts]
                constraint_offset = int(problem.data.vio.numpy()[0] + problem.data.ccgo.numpy()[0])
                normal_impulses = solver.data.solution.lambdas.numpy()[
                    constraint_offset + 2 : constraint_offset + 3 * count : 3
                ]
                ground_contacts = np.any(contact_bodies == -1, axis=1)
                self.assertAlmostEqual(float(np.sum(normal_impulses[ground_contacts])), 5.0 * 9.81e-3, delta=1.0e-3)
                self.assertAlmostEqual(float(np.sum(normal_impulses)), 15.0 * 9.81e-3, delta=3.0e-3)
                limit_offset = int(problem.data.lcgo.numpy()[0])
                limit_impulse = float(solver.data.solution.lambdas.numpy()[limit_offset])
                limit_velocity = float(solver.data.solution.v_plus.numpy()[limit_offset])
                self.assertGreater(limit_impulse, 1.0e-4)
                self.assertGreaterEqual(limit_velocity, -solver.config[0].tolerance)
                self.assertLessEqual(abs(limit_impulse * limit_velocity), solver.config[0].tolerance)

    def test_06_dvi_warmstart_modes(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)

        internal_solver = _solve_dvi(
            model,
            problem,
            warmstart=WarmStartMode.INTERNAL,
            setup=SimpleNamespace(data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians),
        )
        cold_iterations = int(internal_solver.data.status.numpy()[0]["iterations"])
        _assert_solver_status_converged(self, internal_solver)

        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)
        internal_solver.warmstart(problem, model, data, limits, detector.contacts)
        internal_solver.solve(problem)
        _assert_solver_status_converged(self, internal_solver)
        self.assertLessEqual(int(internal_solver.data.status.numpy()[0]["iterations"]), cold_iterations)

        unpack_constraint_solutions(
            lambdas=internal_solver.data.solution.lambdas,
            v_plus=internal_solver.data.solution.v_plus,
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
        )
        container_solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                max_alternating_iterations=300,
                inequality_sweeps_per_iteration=1,
                tolerance=1e-4,
                regularization=1e-5,
            ),
            warmstart=WarmStartMode.CONTAINERS,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)
        container_solver.warmstart(problem, model, data, limits, detector.contacts)
        container_solver.solve(problem)
        _assert_solver_status_converged(self, container_solver)
        _check_solution_matches_dual_problem(self, problem, container_solver)

    def test_06a_dvi_masked_reset_preserves_unselected_worlds(self):
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=3,
            build_fn=basics.build_box_on_plane,
            ground=True,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(
            model,
            problem,
            setup=SimpleNamespace(data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians),
        )
        lambdas_before = solver.data.solution.lambdas.numpy().copy()
        v_plus_before = solver.data.solution.v_plus.numpy().copy()

        world_mask = wp.array([False, True, False], dtype=wp.bool, device=self.device)
        solver.reset(problem=problem, world_mask=world_mask)

        lambdas_after = extract_problem_vector(
            problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=False
        )
        v_plus_after = extract_problem_vector(
            problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=False
        )
        lambdas_before = extract_problem_vector(problem.delassus, lambdas_before, only_active_dims=False)
        v_plus_before = extract_problem_vector(problem.delassus, v_plus_before, only_active_dims=False)
        np.testing.assert_array_equal(lambdas_after[0], lambdas_before[0])
        np.testing.assert_array_equal(lambdas_after[2], lambdas_before[2])
        np.testing.assert_array_equal(v_plus_after[0], v_plus_before[0])
        np.testing.assert_array_equal(v_plus_after[2], v_plus_before[2])
        np.testing.assert_array_equal(lambdas_after[1], np.zeros_like(lambdas_after[1]))
        np.testing.assert_array_equal(v_plus_after[1], np.zeros_like(v_plus_after[1]))

    def test_12_dvi_opening_contact_releases_warmstarted_force(self):
        radius = 0.1
        separation = 0.005
        gap = 0.03
        z = radius + separation

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(gap=gap, margin=0.0)
        body = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_sphere(body=body, radius=radius, cfg=shape_cfg)
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
            limit_lower=-10.0,
            limit_upper=10.0,
        )
        builder.add_articulation([joint])
        builder.add_ground_plane(cfg=shape_cfg)
        model = builder.finalize(device=self.device)

        joint_qd = model.joint_qd.numpy()
        joint_qd[:] = 1.0
        model.joint_qd.assign(joint_qd)

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        config = SolverKamino.Config(
            use_collision_detector=True,
            collision_detector=kamino_config.CollisionDetectorConfig(
                max_contacts_per_world=8,
                max_contacts_per_pair=8,
                default_gap=gap,
            ),
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(
                tolerance=1e-5,
                regularization=1e-5,
                max_alternating_iterations=32,
                inequality_sweeps_per_iteration=4,
            ),
        )
        solver = SolverKamino(model, config=config)

        solver.step(state_0, state_1, control=None, contacts=None, dt=1e-3)
        self.assertEqual(int(solver._contacts_kamino.model_active_contacts.numpy()[0]), 1)

        cache = solver._solver_kamino._ws_contacts.cache
        self.assertIsNotNone(cache)
        reaction = cache.reaction.numpy()
        reaction[:, :] = 0.0
        reaction[0, 2] = 10000.0
        cache.reaction.assign(reaction)
        velocity = cache.velocity.numpy()
        velocity[:, :] = 0.0
        cache.velocity.assign(velocity)

        solver.step(state_1, state_0, control=None, contacts=None, dt=1e-3)

        contact_count = int(solver._contacts_kamino.model_active_contacts.numpy()[0])
        gaps = solver._contacts_kamino.gapfunc.numpy()[:contact_count, 3]
        contact_velocity = solver._contacts_kamino.velocity.numpy()[:contact_count, 2]
        contact_reaction = solver._contacts_kamino.reaction.numpy()[:contact_count, 2]
        opening = (gaps > 0.0) & (contact_velocity > 0.0)

        self.assertTrue(np.any(opening))
        self.assertLess(float(np.max(np.abs(contact_reaction[opening]))), 1e-3)
        self.assertLess(float(abs(state_0.body_qd.numpy()[0, 2])), 2.0)
        self.assertEqual(int(solver._solver_kamino.solver_fd.data.status.numpy()[0]["converged"]), 1)

    def test_03h_dvi_canonical_contact_solution_metrics(self):
        for builder_fn, max_world_contacts in (
            (basics.build_box_on_plane, 4),
            (basics.build_boxes_hinged, 8),
        ):
            for sparse in (False, True):
                with self.subTest(builder=builder_fn.__name__, sparse=sparse):
                    test = TestSetup(
                        builder_fn=builder_fn,
                        max_world_contacts=max_world_contacts,
                        gravity=True,
                        perturb=True,
                        device=self.device,
                        sparse=sparse,
                    )
                    test.build()
                    config = SolverKamino.Config(
                        dynamics_solver="dvi",
                        sparse_dynamics=sparse,
                        sparse_jacobian=sparse,
                    ).dvi
                    solver = _solve_dvi(test.model, test.problem, config=config, setup=test)
                    solution_metrics = _evaluate_solution_metrics(test, solver)

                    _assert_solution_finite(self, solver)
                    for name, value in solution_metrics.items():
                        self.assertTrue(np.isfinite(value), msg=f"{name}={value}")

                    # DVI trades some contact accuracy for throughput, but its
                    # solution must still satisfy dynamics and cone feasibility.
                    self.assertLess(solution_metrics["r_eom"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_kinematics"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_cts_joints"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_cts_contacts"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_v_plus"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_primal"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_dual"], 1.0e-2, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_compl"], 1.0e-2, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_vi_natmap"], 1.0e-2, msg=str(solution_metrics))

    def test_08c_dvi_zero_friction_preserves_tangent_momentum(self):
        """Preserve horizontal momentum exactly when Coulomb friction is zero."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, gap=0.0, margin=0.0)
        body = builder.add_link(
            xform=wp.transformf((0.0, 0.0, 0.1), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
        joint = builder.add_joint_free(parent=-1, child=body)
        builder.add_articulation([joint])
        builder.add_ground_plane(cfg=shape_cfg)
        model = builder.finalize(device=self.device)

        initial_speed = 3.0
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    use_collision_detector=True,
                    sparse_dynamics=sparse,
                    sparse_jacobian=sparse,
                    collision_detector=kamino_config.CollisionDetectorConfig(
                        max_contacts=16,
                        max_contacts_per_world=16,
                        max_contacts_per_pair=8,
                    ),
                )
                solver = SolverKamino(model, config=config)
                state_0 = model.state()
                state_1 = model.state()
                joint_qd = state_0.joint_qd.numpy()
                joint_qd[0] = initial_speed
                state_0.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

                velocities = []
                max_tangent_impulse = 0.0
                contact_seen = False
                for _ in range(50):
                    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-3)
                    state_0, state_1 = state_1, state_0
                    velocities.append(float(state_0.body_qd.numpy()[body, 0]))

                    solver_kamino = solver._solver_kamino
                    problem = solver_kamino._problem_fd
                    solver_dvi = solver_kamino.solver_fd
                    contact_count = int(solver._contacts_kamino.world_active_contacts.numpy()[0])
                    contact_seen = contact_seen or contact_count > 0
                    constraint_offset = int(problem.data.vio.numpy()[0] + problem.data.ccgo.numpy()[0])
                    contact_lambdas = solver_dvi.data.solution.lambdas.numpy()[
                        constraint_offset : constraint_offset + 3 * contact_count
                    ].reshape((-1, 3))
                    if contact_count > 0:
                        max_tangent_impulse = max(
                            max_tangent_impulse,
                            float(np.max(np.abs(contact_lambdas[:, :2]))),
                        )

                self.assertTrue(contact_seen)
                self.assertGreater(int(solver_dvi.data.state.inequality_num_colors.numpy()[0]), 0)
                np.testing.assert_allclose(velocities, initial_speed, rtol=0.0, atol=1.0e-6)
                self.assertLessEqual(max_tangent_impulse, 1.0e-8)

    def test_08d_dvi_kinetic_friction_matches_coulomb_deceleration(self):
        """Match analytic Coulomb deceleration for a sliding box."""
        friction = 0.5
        initial_speed = 3.0
        dt = 2.0e-3
        steps = 50

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(mu=friction, gap=0.0, margin=0.0)
        body = builder.add_link(
            xform=wp.transformf((0.0, 0.0, 0.1), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
        joint = builder.add_joint_free(parent=-1, child=body)
        builder.add_articulation([joint])
        builder.add_ground_plane(cfg=shape_cfg)
        model = builder.finalize(device=self.device)

        expected_speeds = initial_speed - friction * 9.81 * dt * np.arange(1, steps + 1)
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    use_collision_detector=True,
                    sparse_dynamics=sparse,
                    sparse_jacobian=sparse,
                    collision_detector=kamino_config.CollisionDetectorConfig(
                        max_contacts=16,
                        max_contacts_per_world=16,
                        max_contacts_per_pair=8,
                    ),
                )
                solver = SolverKamino(model, config=config)
                state_0 = model.state()
                state_1 = model.state()
                joint_qd = state_0.joint_qd.numpy()
                joint_qd[0] = initial_speed
                state_0.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

                measured_speeds = []
                for _ in range(steps):
                    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                    state_0, state_1 = state_1, state_0
                    measured_speeds.append(float(state_0.body_qd.numpy()[body, 0]))

                self.assertGreater(int(solver._contacts_kamino.world_active_contacts.numpy()[0]), 0)
                np.testing.assert_allclose(measured_speeds, expected_speeds, rtol=0.0, atol=1.0e-4)

    def test_08d2_dvi_friction_sweep_matches_speed_and_distance(self):
        """Match analytic sliding speed and distance across friction values."""
        initial_speed = 3.0
        dt = 2.0e-3
        steps = 50

        for friction in (0.1, 0.3, 0.8):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            SolverKamino.register_custom_attributes(builder)
            shape_cfg = newton.ModelBuilder.ShapeConfig(mu=friction, gap=0.0, margin=0.0)
            body = builder.add_link(
                xform=wp.transformf((0.0, 0.0, 0.1), wp.quat_identity()),
                mass=1.0,
            )
            builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
            joint = builder.add_joint_free(parent=-1, child=body)
            builder.add_articulation([joint])
            builder.add_ground_plane(cfg=shape_cfg)
            model = builder.finalize(device=self.device)

            expected_speed = initial_speed - friction * 9.81 * dt * steps
            expected_distance = initial_speed * dt * steps - friction * 9.81 * dt * dt * steps * (steps + 1) / 2.0
            for sparse in (False, True):
                with self.subTest(friction=friction, sparse=sparse):
                    config = SolverKamino.Config(
                        dynamics_solver="dvi",
                        use_collision_detector=True,
                        sparse_dynamics=sparse,
                        sparse_jacobian=sparse,
                        collision_detector=kamino_config.CollisionDetectorConfig(
                            max_contacts=16,
                            max_contacts_per_world=16,
                            max_contacts_per_pair=8,
                        ),
                    )
                    solver = SolverKamino(model, config=config)
                    state_0 = model.state()
                    state_1 = model.state()
                    joint_qd = state_0.joint_qd.numpy()
                    joint_qd[0] = initial_speed
                    state_0.joint_qd.assign(joint_qd)
                    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

                    for _ in range(steps):
                        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                        state_0, state_1 = state_1, state_0

                    self.assertGreater(int(solver._contacts_kamino.world_active_contacts.numpy()[0]), 0)
                    self.assertAlmostEqual(
                        float(state_0.body_qd.numpy()[body, 0]),
                        expected_speed,
                        delta=1.0e-4,
                    )
                    self.assertAlmostEqual(
                        float(state_0.body_q.numpy()[body, 0]),
                        expected_distance,
                        delta=1.0e-5,
                    )

    def test_08d3_dvi_friction_propagates_through_fixed_joint(self):
        """Converge articulated contact friction at a 2 kHz step rate."""
        friction = 0.2
        initial_speed = 3.0
        dt = 5.0e-4
        steps = 200

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(mu=friction, gap=0.0, margin=0.0)
        contact_body = builder.add_link(
            xform=wp.transformf((0.0, 0.0, 0.1), wp.quat_identity()),
            mass=1.0,
        )
        carried_body = builder.add_link(
            xform=wp.transformf((0.0, 0.0, 0.2), wp.quat_identity()),
            mass=9.0,
            inertia=wp.mat33(np.eye(3) * 0.06),
        )
        builder.add_shape_box(body=contact_body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
        root_joint = builder.add_joint_free(parent=-1, child=contact_body)
        fixed_joint = builder.add_joint_fixed(
            parent=contact_body,
            child=carried_body,
            parent_xform=wp.transformf((0.0, 0.0, 0.1), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([root_joint, fixed_joint])
        builder.add_ground_plane(cfg=shape_cfg)
        model = builder.finalize(device=self.device)

        expected_speed = initial_speed - friction * 9.81 * dt * steps
        expected_distance = initial_speed * dt * steps - friction * 9.81 * dt * dt * steps * (steps + 1) / 2.0
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    use_collision_detector=True,
                    sparse_dynamics=sparse,
                    sparse_jacobian=sparse,
                    collision_detector=kamino_config.CollisionDetectorConfig(
                        max_contacts=16,
                        max_contacts_per_world=16,
                        max_contacts_per_pair=8,
                    ),
                )
                solver = SolverKamino(model, config=config)
                state_0 = model.state()
                state_1 = model.state()
                joint_qd = state_0.joint_qd.numpy()
                joint_qd[0] = initial_speed
                state_0.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

                solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                state_0, state_1 = state_1, state_0
                dvi_solver = solver._solver_kamino.solver_fd
                first_sweeps = int(dvi_solver.data.status.numpy()[0]["iterations"])
                _assert_solver_status_converged(self, dvi_solver)
                for _ in range(1, steps):
                    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                    state_0, state_1 = state_1, state_0

                _assert_solver_status_converged(self, dvi_solver)
                self.assertGreater(first_sweeps, 1)
                positions = state_0.body_q.numpy()
                velocities = state_0.body_qd.numpy()
                self.assertGreater(int(solver._contacts_kamino.world_active_contacts.numpy()[0]), 0)
                np.testing.assert_allclose(
                    velocities[[contact_body, carried_body], 0],
                    expected_speed,
                    rtol=0.0,
                    atol=2.0e-4,
                )
                np.testing.assert_allclose(
                    positions[[contact_body, carried_body], 0],
                    expected_distance,
                    rtol=0.0,
                    atol=2.0e-5,
                )
                self.assertAlmostEqual(
                    float(positions[carried_body, 2] - positions[contact_body, 2]),
                    0.1,
                    delta=1.0e-5,
                )
                self.assertLess(
                    float(np.linalg.norm(velocities[carried_body] - velocities[contact_body])),
                    1.0e-4,
                )

    def test_08d4_dvi_incline_friction_threshold(self):
        """Hold or slide according to the Coulomb incline threshold."""
        dt = 2.0e-3
        steps = 100
        cases = (
            (25.0, 0.7, False),
            (35.0, 0.3, True),
        )

        for angle_degrees, friction, should_slide in cases:
            angle = float(np.deg2rad(angle_degrees))
            gravity = (9.81 * np.sin(angle), 0.0, -9.81 * np.cos(angle))
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=gravity)
            SolverKamino.register_custom_attributes(builder)
            shape_cfg = newton.ModelBuilder.ShapeConfig(mu=friction, gap=0.0, margin=0.0)
            body = builder.add_link(
                xform=wp.transformf((0.0, 0.0, 0.1), wp.quat_identity()),
                mass=1.0,
            )
            builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
            joint = builder.add_joint_free(parent=-1, child=body)
            builder.add_articulation([joint])
            builder.add_ground_plane(cfg=shape_cfg)
            model = builder.finalize(device=self.device)

            for sparse in (False, True):
                with self.subTest(angle=angle_degrees, friction=friction, sparse=sparse):
                    config = SolverKamino.Config(
                        dynamics_solver="dvi",
                        use_collision_detector=True,
                        sparse_dynamics=sparse,
                        sparse_jacobian=sparse,
                        collision_detector=kamino_config.CollisionDetectorConfig(
                            max_contacts=16,
                            max_contacts_per_world=16,
                            max_contacts_per_pair=8,
                        ),
                    )
                    solver = SolverKamino(model, config=config)
                    state_0 = model.state()
                    state_1 = model.state()
                    for _ in range(steps):
                        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                        state_0, state_1 = state_1, state_0

                    position_x = float(state_0.body_q.numpy()[body, 0])
                    velocity_x = float(state_0.body_qd.numpy()[body, 0])
                    if should_slide:
                        acceleration = 9.81 * (np.sin(angle) - friction * np.cos(angle))
                        expected_velocity = acceleration * dt * steps
                        expected_distance = acceleration * dt * dt * steps * (steps + 1) / 2.0
                        self.assertAlmostEqual(velocity_x, expected_velocity, delta=2.0e-3)
                        self.assertAlmostEqual(position_x, expected_distance, delta=2.0e-4)
                    else:
                        self.assertLess(abs(velocity_x), 1.0e-4)
                        self.assertLess(abs(position_x), 1.0e-5)

    def test_08e_dvi_sliding_sphere_settles_into_analytic_rolling(self):
        """Match the analytic sliding-to-rolling transition of a sphere.

        A sphere contacts the ground through a lever arm, so its tangential
        Delassus diagonal exceeds its normal diagonal. Preconditioning the
        tangential rows by the smaller normal diagonal over-relaxes them, which
        makes the friction impulse oscillate instead of settling: the sphere
        keeps a residual slip and its speed stops decreasing monotonically.
        """
        friction = 0.5
        initial_speed = 2.0
        radius = 0.1
        dt = 1.0e-3
        # Pure sliding reaches rolling at t = 2 * v0 / (7 * mu * g) for a solid
        # sphere; twice that leaves margin for the discrete friction impulse.
        steps = int(2.0 * (2.0 * initial_speed / (7.0 * friction * 9.81)) / dt)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(mu=friction, gap=0.0, margin=0.0)
        body = builder.add_link(xform=wp.transformf((0.0, 0.0, radius), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(body=body, radius=radius, cfg=shape_cfg)
        joint = builder.add_joint_free(parent=-1, child=body)
        builder.add_articulation([joint])
        builder.add_ground_plane(cfg=shape_cfg)
        model = builder.finalize(device=self.device)

        # Rolling without slip conserves v0 = J * (1 / m + r^2 / I), so the
        # terminal speed follows from the finalized mass distribution.
        mass = float(model.body_mass.numpy()[body])
        inertia_yy = float(model.body_inertia.numpy()[body][1, 1])
        expected_rolling_speed = initial_speed / (1.0 + inertia_yy / (mass * radius * radius))

        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    use_collision_detector=True,
                    sparse_dynamics=sparse,
                    sparse_jacobian=sparse,
                    collision_detector=kamino_config.CollisionDetectorConfig(
                        max_contacts=16,
                        max_contacts_per_world=16,
                        max_contacts_per_pair=8,
                    ),
                )
                solver = SolverKamino(model, config=config)
                state_0 = model.state()
                state_1 = model.state()
                joint_qd = state_0.joint_qd.numpy()
                joint_qd[0] = initial_speed
                state_0.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

                speeds = []
                spins = []
                for _ in range(steps):
                    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                    state_0, state_1 = state_1, state_0
                    body_qd = state_0.body_qd.numpy()[body]
                    speeds.append(float(body_qd[0]))
                    spins.append(float(body_qd[4]))

                speeds = np.array(speeds)
                slip = speeds - radius * np.array(spins)
                self.assertAlmostEqual(float(speeds[-1]), expected_rolling_speed, delta=1.0e-4)
                self.assertLessEqual(float(np.abs(slip[-1])), 1.0e-5)
                # Coulomb friction only opposes sliding, so the speed never rises.
                self.assertLessEqual(float(np.max(np.diff(speeds))), 1.0e-5)

    def test_08b_dr_legs_contact_capacity_scales_with_world_count(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs multi-world capacity regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        world_count = 3
        args = SimpleNamespace(
            world_count=world_count,
            use_kamino_contacts=False,
            dynamics_solver="dvi",
        )
        example = Example(ViewerNull(num_frames=1), args)

        expected_capacity = 72 * world_count
        self.assertEqual(example.model.rigid_contact_max, expected_capacity)
        self.assertEqual(example.contacts.rigid_contact_max, expected_capacity)
        self.assertEqual(example.collision_pipeline.rigid_contact_max, expected_capacity)

    def test_09_dr_legs_dvi_first_contact_remains_finite(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI first-contact regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
        )
        example = Example(ViewerNull(num_frames=1), args)

        contact_seen = False
        color_checked = False
        for _ in range(12):
            example.step()
            body_q = example.state_0.body_q.numpy()
            body_qd = example.state_0.body_qd.numpy()
            lambdas = example.solver._solver_kamino.solver_fd.data.solution.lambdas.numpy()
            kamino_contacts = example.solver._contacts_kamino
            contact_count = int(kamino_contacts.world_active_contacts.numpy()[0])
            contact_seen = contact_seen or contact_count > 0
            if contact_count > 0 and not example.config.sparse_dynamics:
                solver_fd = example.solver._solver_kamino.solver_fd
                color_count = int(solver_fd.data.state.inequality_num_colors.numpy()[0])
                colors = solver_fd.data.state.inequality_colors.numpy()
                bid_ab = kamino_contacts.bid_AB.numpy()
                self.assertGreater(color_count, 0)
                self.assertTrue(np.all(colors[:contact_count] >= 0))
                for ci in range(contact_count):
                    bodies_i = {int(bid_ab[ci][0]), int(bid_ab[ci][1])} - {-1}
                    for cj in range(ci):
                        if colors[ci] == colors[cj]:
                            bodies_j = {int(bid_ab[cj][0]), int(bid_ab[cj][1])} - {-1}
                            self.assertFalse(bodies_i & bodies_j)
                color_checked = True
            elif contact_count > 0:
                color_checked = True

            self.assertTrue(np.all(np.isfinite(body_q)))
            self.assertTrue(np.all(np.isfinite(body_qd)))
            self.assertTrue(np.all(np.isfinite(lambdas)))
            self.assertLess(float(np.max(np.abs(body_qd))), 100.0)
            self.assertLess(float(np.max(np.abs(lambdas))), 100.0)

        self.assertTrue(contact_seen)
        self.assertTrue(color_checked)

    def test_10_dr_legs_dvi_tipped_contact_does_not_creep(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI tipped-contact regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
        )
        example = Example(ViewerNull(num_frames=1), args)

        q_tip = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.pi * 0.5))
        example.base_q.assign([wp.transformf((0.0, 0.0, 0.25), q_tip)])
        reset_config = SolverKamino.ResetConfig(base_pose=SolverKamino.ResetConfig.FromBaseQ(example.base_q))
        example.solver.reset(state=example.state_0, config=reset_config)
        example.solver.reset(state=example.state_1, config=reset_config)
        example.capture()

        base_start = example.state_0.body_q.numpy()[0, :3].copy()
        contact_seen = False
        post_settle_penetration = []
        post_settle_xy = []
        for step_idx in range(400):
            example.step()
            contact_seen = contact_seen or int(example.contacts.rigid_contact_count.numpy()[0]) > 0
            contacts_kamino = example.solver._contacts_kamino
            contact_count = int(contacts_kamino.world_active_contacts.numpy()[0])
            if step_idx >= 40 and contact_count > 0:
                gaps = contacts_kamino.gapfunc.numpy()[:contact_count, 3]
                post_settle_penetration.append(float(max(0.0, -np.min(gaps))))
            if step_idx >= 200:
                post_settle_xy.append(example.state_0.body_q.numpy()[0, :2].copy())

        body_q = example.state_0.body_q.numpy()
        body_qd = example.state_0.body_qd.numpy()
        base_delta_xy = body_q[0, :2] - base_start[:2]

        self.assertTrue(contact_seen)
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(body_qd)))
        self.assertLess(float(np.linalg.norm(base_delta_xy)), 0.008)
        self.assertGreater(len(post_settle_penetration), 0)
        self.assertLess(float(np.percentile(post_settle_penetration, 95)), 0.0035)
        self.assertLess(float(np.linalg.norm(post_settle_xy[-1] - post_settle_xy[0])), 2.0e-4)

    def test_11_dr_legs_dvi_contact_force_balances_weight(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI contact-force regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton._src.solvers.kamino._src.geometry.aggregation import ContactAggregation  # noqa: PLC0415
        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
        )
        example = Example(ViewerNull(num_frames=1), args)

        base_z = []
        for _ in range(180):
            example.step()
            base_z.append(float(example.state_0.body_q.numpy()[0, 2]))

        contacts_kamino = example.solver._contacts_kamino
        aggregation = ContactAggregation(model=example.solver._model_kamino, contacts=contacts_kamino)
        aggregation.compute()

        contact_count = int(contacts_kamino.world_active_contacts.numpy()[0])
        total_contact_force = aggregation.body_net_force.numpy()[0].sum(axis=0)
        weight = float(example.model.body_mass.numpy().sum() * 9.81)
        force_ratio = float(total_contact_force[2] / weight)

        self.assertGreater(contact_count, 0)
        self.assertTrue(np.all(np.isfinite(total_contact_force)))
        self.assertGreater(force_ratio, 0.95)
        self.assertLess(force_ratio, 1.05)
        z = np.array(base_z[60:], dtype=np.float64)
        x = np.arange(z.size, dtype=np.float64)
        residual = z - np.polyval(np.polyfit(x, z, 1), x)
        self.assertLess(float(np.max(residual) - np.min(residual)), 0.001)


if __name__ == "__main__":
    unittest.main()
