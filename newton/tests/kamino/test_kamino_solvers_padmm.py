# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Proximal-ADMM Solver."""

import unittest
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.kinematics.constraints import unpack_constraint_solutions
from newton._src.solvers.kamino._src.linalg import ConjugateResidualSolver, LLTBlockedSolver
from newton._src.solvers.kamino._src.linalg.utils.matrix import SquareSymmetricMatrixProperties
from newton._src.solvers.kamino._src.linalg.utils.range import in_range_via_gaussian_elimination
from newton._src.solvers.kamino._src.models.builders import basics
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver, PADMMWarmStartMode
from newton._src.solvers.kamino._src.solvers.padmm.math import (
    project_to_coulomb_cone,
    project_to_coulomb_dual_cone,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import (
    extract_delassus,
    extract_info_vectors,
    extract_problem_vector,
)
from newton.tests.kamino.utils.make import make_containers, update_containers

###
# Helper functions
###


class TestSetup:
    def __init__(
        self,
        builder_fn,
        max_world_contacts: int = 32,
        perturb: bool = True,
        gravity: bool = True,
        device: wp.DeviceLike = None,
        sparse: bool = False,
        **kwargs,
    ):
        # Cache the max contacts allocated for the test problem
        self.max_world_contacts = max_world_contacts

        # Construct the model description using model builders for different systems
        self.builder: ModelBuilderKamino = builder_fn(**kwargs)

        # Set ad-hoc configurations
        if not gravity:
            self.builder.set_gravity(wp.vec3f(0.0))
        if perturb:
            u_0 = wp.spatial_vectorf(10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for body in self.builder.all_bodies:
                body.u_i_0 = u_0

        # Create the model and containers from the builder
        self.model, self.data, self.state, self.limits, self.detector, self.jacobians = make_containers(
            builder=self.builder, max_world_contacts=max_world_contacts, device=device, sparse=sparse
        )
        self.contacts = self.detector.contacts
        self.state_p = self.model.state()

        # Create the DualProblem to be solved
        self.problem = DualProblem(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            jacobians=self.jacobians,
            solver=ConjugateResidualSolver if sparse else LLTBlockedSolver,
            sparse=sparse,
        )

        # Update the sim data containers
        update_containers(
            model=self.model,
            data=self.data,
            state=self.state,
            limits=self.limits,
            detector=self.detector,
            jacobians=self.jacobians,
        )

    def build(self):
        # Build the dual problem
        self.problem.build(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            jacobians=self.jacobians,
        )

    def cache(self, solver: PADMMSolver):
        # Unpack the computed constraint multipliers to the respective joint-limit
        # and contact data for post-processing and optional solver warm-starting
        unpack_constraint_solutions(
            lambdas=solver.data.solution.lambdas,
            v_plus=solver.data.solution.v_plus,
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
        )


def print_dual_problem_summary(D: np.ndarray, v_f: np.ndarray, notes: str = ""):
    D_props = SquareSymmetricMatrixProperties(D)
    v_f_is_in_range, *_ = in_range_via_gaussian_elimination(D, v_f)
    msg.info("Delassus Properties %s:\n%s\n", notes, D_props)
    msg.info("v_f is in range of D %s: %s\n", notes, v_f_is_in_range)


def check_padmm_solution(
    test: unittest.TestCase, model: ModelKamino, problem: DualProblem, solver: PADMMSolver, verbose: bool = False
):
    # Extract numpy arrays from the solver state and solution
    only_active_dims = True
    D_wp_np = extract_delassus(problem.delassus, only_active_dims=only_active_dims)
    v_f_wp_np = extract_problem_vector(problem.delassus, problem.data.v_f.numpy(), only_active_dims=only_active_dims)
    P_wp_np = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=only_active_dims)
    v_plus_wp_np = extract_problem_vector(
        problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=only_active_dims
    )
    lambdas_wp_np = extract_problem_vector(
        problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=only_active_dims
    )

    # Optional verbose output
    status = solver.data.status.numpy()
    for w in range(model.size.num_worlds):
        # Recover the original (preconditioned) Delassua matrix from the in-place regularized storage
        dtype = D_wp_np[w].dtype
        ncts = D_wp_np[w].shape[0]
        I_np = dtype.type(solver.config[0].eta + solver.config[0].rho_0) * np.eye(D_wp_np[w].shape[0], dtype=dtype)
        D = D_wp_np[w] - I_np

        # Recover original Delassus matrix and v_f from preconditioned versions
        D_true = np.diag(np.reciprocal(P_wp_np[w])) @ D @ np.diag(np.reciprocal(P_wp_np[w]))
        v_f_true = np.diag(np.reciprocal(P_wp_np[w])) @ v_f_wp_np[w]

        # Compute the true dual solution and error
        v_plus_true = np.matmul(D_true, lambdas_wp_np[w]) + v_f_true
        error_dual_abs_l2 = np.linalg.norm(v_plus_true - v_plus_wp_np[w]) / float(ncts)
        error_dual_abs_inf = np.linalg.norm(v_plus_true - v_plus_wp_np[w], ord=np.inf)

        # Extract solver status
        converged = True if status[w][0] == 1 else False
        iterations = status[w][1]
        r_p = status[w][2]
        r_d = status[w][3]
        r_c = status[w][4]

        # Optionally print relevant solver data
        if verbose:
            print_dual_problem_summary(D, v_f_wp_np[w], "(preconditioned)")
            print_dual_problem_summary(D_true, v_f_true)
            msg.notif(
                "\n---------"
                f"\nconverged: {converged}"
                f"\niterations: {iterations}"
                "\n---------"
                f"\nr_p: {r_p}"
                f"\nr_d: {r_d}"
                f"\nr_c: {r_c}"
                "\n---------"
                f"\nerror_dual_abs_l2: {error_dual_abs_l2}"
                f"\nerror_dual_abs_inf: {error_dual_abs_inf}"
                "\n---------"
                f"\nsolution: lambda: {lambdas_wp_np[w]}"
                f"\nsolution: v_plus: {v_plus_wp_np[w]}"
                "\n---------\n"
            )

        # Check results
        test.assertTrue(converged)
        test.assertLessEqual(iterations, solver.config[w].max_iterations)
        test.assertLessEqual(r_p, solver.config[w].primal_tolerance)
        test.assertLessEqual(r_d, solver.config[w].dual_tolerance)
        test.assertLessEqual(r_c, solver.config[w].compl_tolerance)
        # Using expanded tolerance for true dual error due to potential numerical inaccuracies
        # between in-solver residual and true residual.
        test.assertLessEqual(error_dual_abs_l2, solver.config[w].dual_tolerance * 4.0)
        test.assertLessEqual(error_dual_abs_inf, solver.config[w].dual_tolerance * 4.0)


def solve_apadmm_bilateral_reference(
    D: np.ndarray,
    v_f: np.ndarray,
    P: np.ndarray,
    config: PADMMSolver.Config,
) -> dict[str, Any]:
    """Solve a bilateral APADMM problem with a NumPy iteration reference.

    This mirrors the accelerated PADMM state transitions for one dense,
    preconditioned bilateral problem. The bilateral feasible set is all of
    :math:`R^n`, so the production solver's projection is the identity:
    ``y = x - z_hat / rho``. It consequently does not model unilateral limit
    or contact projection, De Saxce corrections, or complementarity residuals.

    The reference also omits production features that are irrelevant to this
    fixed-iteration regression: multiple worlds, sparse/iterative linear
    solves, adaptive penalty updates, and convergence-based early termination.
    It instead solves the fixed dense system with NumPy and executes exactly
    ``max_iterations`` iterations, recording the state-derived quantities
    exposed through ``PADMMInfo`` for comparison.
    """
    dtype = D.dtype
    eta = dtype.type(config.eta)
    rho = dtype.type(config.rho_0)
    restart_tolerance = dtype.type(config.restart_tolerance)
    a_0 = dtype.type(config.a_0)
    lhs = D + (eta + rho) * np.eye(D.shape[0], dtype=dtype)

    x = np.zeros_like(v_f)
    y = np.zeros_like(v_f)
    z = np.zeros_like(v_f)
    x_p = np.zeros_like(v_f)
    y_p = np.zeros_like(v_f)
    z_p = np.zeros_like(v_f)
    y_hat = np.zeros_like(v_f)
    z_hat = np.zeros_like(v_f)
    a_p = a_0
    r_a_p = dtype.type(np.finfo(dtype).max)
    restart = 0
    num_restarts = 0
    history = {
        "a": [],
        "norm_x": [],
        "norm_y": [],
        "norm_z": [],
        "r_dx": [],
        "r_dy": [],
        "r_dz": [],
        "r_comb": [],
        "r_comb_ratio": [],
        "num_restarts": [],
    }

    for _ in range(config.max_iterations):
        x_candidate = np.linalg.solve(lhs, -v_f + eta * x_p + rho * y_hat + z_hat)
        y_candidate = x_candidate - z_hat / rho
        z_candidate = z_hat + rho * (y_candidate - x_candidate)

        dx = P * (x_candidate - x_p)
        dy = P * (y_candidate - y_hat)
        dz = (z_candidate - z_hat) / P
        r_a = rho * np.dot(dy, dy) + np.dot(dz, dz) / rho

        if r_a < restart_tolerance * r_a_p:
            restart = 0
            a = dtype.type((1.0 + np.sqrt(1.0 + 4.0 * a_p * a_p)) / 2.0)
            factor = dtype.type((a_p - 1.0) / a)
            x = x_candidate
            y = y_candidate
            z = z_candidate
            y_hat = y + factor * (y - y_p)
            z_hat = z + factor * (z - z_p)
            x_p = x.copy()
            y_p = y.copy()
            z_p = z.copy()
        else:
            # A restart drops the momentum but keeps the current iterate.
            restart = 1
            num_restarts += 1
            r_a = r_a_p / restart_tolerance
            a = a_0
            y_hat = y_p.copy()
            z_hat = z_p.copy()
            x = x_candidate
            y = y_candidate
            z = z_candidate
            x_p = x.copy()
            y_p = y.copy()
            z_p = z.copy()

        a_p = a
        r_a_pp = r_a_p
        r_a_p = dtype.type(r_a)

        history["a"].append(a)
        history["norm_x"].append(np.linalg.norm(x))
        history["norm_y"].append(np.linalg.norm(y))
        history["norm_z"].append(np.linalg.norm(z))
        history["r_dx"].append(np.linalg.norm(dx))
        history["r_dy"].append(np.linalg.norm(dy))
        history["r_dz"].append(np.linalg.norm(dz))
        history["r_comb"].append(r_a)
        history["r_comb_ratio"].append(r_a / r_a_pp)
        history["num_restarts"].append(num_restarts)

    return {
        "x": x,
        "y": y,
        "z": z,
        "x_p": x_p,
        "y_p": y_p,
        "z_p": z_p,
        "y_hat": y_hat,
        "z_hat": z_hat,
        "restart": restart,
        "num_restarts": num_restarts,
        "history": {key: np.asarray(values, dtype=dtype) for key, values in history.items()},
    }


def save_solver_info(solver: PADMMSolver, path: str | None = None, verbose: bool = False):
    # Attempt to import matplotlib for plotting
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib is not available so we skip plotting

    solver_has_acceleration = solver._use_acceleration

    nw = solver.size.num_worlds
    status = solver.data.status.numpy()
    iterations = [status[w][1] for w in range(nw)]
    offsets_np = solver.data.info.offsets.numpy()
    num_rho_updates_np = extract_info_vectors(offsets_np, solver.data.info.num_rho_updates.numpy(), iterations)
    norm_s_np = extract_info_vectors(offsets_np, solver.data.info.norm_s.numpy(), iterations)
    norm_x_np = extract_info_vectors(offsets_np, solver.data.info.norm_x.numpy(), iterations)
    norm_y_np = extract_info_vectors(offsets_np, solver.data.info.norm_y.numpy(), iterations)
    norm_z_np = extract_info_vectors(offsets_np, solver.data.info.norm_z.numpy(), iterations)
    f_ccp_np = extract_info_vectors(offsets_np, solver.data.info.f_ccp.numpy(), iterations)
    f_ncp_np = extract_info_vectors(offsets_np, solver.data.info.f_ncp.numpy(), iterations)
    r_dx_np = extract_info_vectors(offsets_np, solver.data.info.r_dx.numpy(), iterations)
    r_dy_np = extract_info_vectors(offsets_np, solver.data.info.r_dy.numpy(), iterations)
    r_dz_np = extract_info_vectors(offsets_np, solver.data.info.r_dz.numpy(), iterations)
    r_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_primal.numpy(), iterations)
    r_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_dual.numpy(), iterations)
    r_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_compl.numpy(), iterations)
    r_pd_np = extract_info_vectors(offsets_np, solver.data.info.r_pd.numpy(), iterations)
    r_dp_np = extract_info_vectors(offsets_np, solver.data.info.r_dp.numpy(), iterations)
    r_ncp_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_primal.numpy(), iterations)
    r_ncp_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_dual.numpy(), iterations)
    r_ncp_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_compl.numpy(), iterations)
    r_ncp_natmap_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_natmap.numpy(), iterations)

    if solver_has_acceleration:
        num_restarts_np = extract_info_vectors(offsets_np, solver.data.info.num_restarts.numpy(), iterations)
        a_np = extract_info_vectors(offsets_np, solver.data.info.a.numpy(), iterations)
        r_comb_np = extract_info_vectors(offsets_np, solver.data.info.r_comb.numpy(), iterations)
        r_comb_ratio_np = extract_info_vectors(offsets_np, solver.data.info.r_comb_ratio.numpy(), iterations)

    if verbose:
        for w in range(nw):
            print(f"[World {w}] =======================================================================")
            print(f"solver.info.num_rho_updates: {num_rho_updates_np[w]}")
            print(f"solver.info.norm_s: {norm_s_np[w]}")
            print(f"solver.info.norm_x: {norm_x_np[w]}")
            print(f"solver.info.norm_y: {norm_y_np[w]}")
            print(f"solver.info.norm_z: {norm_z_np[w]}")
            print(f"solver.info.f_ccp: {f_ccp_np[w]}")
            print(f"solver.info.f_ncp: {f_ncp_np[w]}")
            print(f"solver.info.r_dx: {r_dx_np[w]}")
            print(f"solver.info.r_dy: {r_dy_np[w]}")
            print(f"solver.info.r_dz: {r_dz_np[w]}")
            print(f"solver.info.r_primal: {r_primal_np[w]}")
            print(f"solver.info.r_dual: {r_dual_np[w]}")
            print(f"solver.info.r_compl: {r_compl_np[w]}")
            print(f"solver.info.r_pd: {r_pd_np[w]}")
            print(f"solver.info.r_dp: {r_dp_np[w]}")
            print(f"solver.info.r_ncp_primal: {r_ncp_primal_np[w]}")
            print(f"solver.info.r_ncp_dual: {r_ncp_dual_np[w]}")
            print(f"solver.info.r_ncp_compl: {r_ncp_compl_np[w]}")
            print(f"solver.info.r_ncp_natmap: {r_ncp_natmap_np[w]}")
            if solver_has_acceleration:
                print(f"solver.info.num_restarts: {num_restarts_np[w]}")
                print(f"solver.info.a: {a_np[w]}")
                print(f"solver.info.r_comb: {r_comb_np[w]}")
                print(f"solver.info.r_comb_ratio: {r_comb_ratio_np[w]}")

    # List of (label, data) for plotting
    info_list = [
        ("num_rho_updates", num_rho_updates_np),
        ("norm_s", norm_s_np),
        ("norm_x", norm_x_np),
        ("norm_y", norm_y_np),
        ("norm_z", norm_z_np),
        ("f_ccp", f_ccp_np),
        ("f_ncp", f_ncp_np),
        ("r_dx", r_dx_np),
        ("r_dy", r_dy_np),
        ("r_dz", r_dz_np),
        ("r_primal", r_primal_np),
        ("r_dual", r_dual_np),
        ("r_compl", r_compl_np),
        ("r_pd", r_pd_np),
        ("r_dp", r_dp_np),
        ("r_ncp_primal", r_ncp_primal_np),
        ("r_ncp_dual", r_ncp_dual_np),
        ("r_ncp_compl", r_ncp_compl_np),
        ("r_ncp_natmap", r_ncp_natmap_np),
    ]
    if solver_has_acceleration:
        info_list.extend(
            [
                ("num_restarts", num_restarts_np),
                ("a", a_np),
                ("r_comb", r_comb_np),
                ("r_comb_ratio", r_comb_ratio_np),
            ]
        )

    # Plot all info as subplots: rows=info_list, cols=worlds
    n_rows = len(info_list)
    n_cols = nw
    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    for row, (label, arr) in enumerate(info_list):
        for col in range(nw):
            ax = axes[row, col]
            ax.plot(arr[col], label=f"{label}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"World {col}")
            if col == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("")
            ax.grid(True)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


###
# Tests
###


class TestPADMMSolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.savefig = test_context.verbose  # Set to True to generate solver info plots
        self.default_device = wp.get_device(test_context.device)
        self.output_path = test_context.output_path / "test_solvers_padmm"

        # Create output directory if saving figures
        if self.savefig:
            self.output_path.mkdir(parents=True, exist_ok=True)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_padmm_default(self):
        """
        Test creating a PADMMSolver with default initialization.
        """
        # Creating a default PADMMSolver without any model or config
        # should result in a solver without any memory allocation.
        solver = PADMMSolver()
        self.assertIsNone(solver._size)
        self.assertIsNone(solver._data)
        self.assertIsNone(solver._device)
        self.assertEqual(solver.config, [])

        # Requesting the solver data container when the
        # solver has not been finalized should raise an
        # error since no allocations have been made.
        self.assertRaises(RuntimeError, lambda: solver.data)

    def test_01_finalize_padmm_default(self):
        """
        Test creating a PADMMSolver with default initialization and then finalizing all memory allocations.
        """
        # Create a test setup
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Creating a default PADMMSolver without any model or config
        # should result in a solver without any memory allocation.
        solver = PADMMSolver()

        # Finalize the solver with a model
        solver.finalize(test.model)

        # Check that the solver has been properly allocated
        self.assertIsNotNone(solver._size)
        self.assertIsNotNone(solver._data)
        self.assertIsNotNone(solver._device)
        self.assertEqual(len(solver.config), test.model.size.num_worlds)
        self.assertIs(solver._device, test.model.device)
        self.assertIs(solver.size, test.model.size)

    def test_02_padmm_solve(self):
        """
        Tests the Proximal-ADMM (PADMM) solver with default config on the reference problem.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the PADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=False,
            collect_info=self.savefig,
        )

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        # check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_02_padmm_solve.pdf"
            save_solver_info(solver=solver, path=str(path))

        # Check solution
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

    def test_03_padmm_solve_with_acceleration(self):
        """
        Tests the Accelerated Proximal-ADMM (APADMM) solver on the reference problem with Nesterov acceleration.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.restart_tolerance = 0.999
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the PADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=True,
            collect_info=self.savefig,
        )

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        # check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_03_padmm_solve_with_acceleration.pdf"
            save_solver_info(solver=solver, path=str(path))

        # Check solution
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

    def test_04_padmm_solve_with_internal_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with internal warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.INTERNAL,
            use_acceleration=False,
            collect_info=self.savefig,
        )

        # Initial cold-started solve
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Second solve with warm-starting from previous solution
        test.build()
        solver.warmstart(test.problem, test.model, test.data)
        expected_forces = solver.data.solution.lambdas.numpy() * config.warmstart_scale
        np.testing.assert_allclose(solver.data.state.x_p.numpy(), expected_forces)
        np.testing.assert_allclose(solver.data.state.y_p.numpy(), expected_forces)
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_04_padmm_solve_with_internal_warmstart.pdf"
            save_solver_info(solver=solver, path=str(path))

    def test_05_padmm_solve_with_container_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.CONTAINERS,
            use_acceleration=False,
            collect_info=self.savefig,
        )

        # Initial cold-started solve
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Second solve with warm-starting from previous solution
        test.cache(solver=solver)
        test.build()
        solver.warmstart(test.problem, test.model, test.data, test.limits, test.contacts)
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_05_padmm_solve_with_container_warmstart.pdf"
            save_solver_info(solver=solver, path=str(path))

    def test_06_padmm_solve_with_acceleration_and_internal_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.restart_tolerance = 0.999
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.INTERNAL,
            use_acceleration=True,
            collect_info=self.savefig,
        )

        # Initial cold-started solve
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Second solve with warm-starting from previous solution
        test.build()
        solver.warmstart(test.problem, test.model, test.data)
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_06_padmm_solve_with_acceleration_and_internal_warmstart.pdf"
            save_solver_info(solver=solver, path=str(path))

    def test_07_padmm_solve_with_acceleration_and_container_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.restart_tolerance = 0.999
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.CONTAINERS,
            use_acceleration=True,
            collect_info=self.savefig,
        )

        # Initial cold-started solve
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Second solve with warm-starting from previous solution
        test.cache(solver=solver)
        test.build()
        solver.warmstart(test.problem, test.model, test.data, test.limits, test.contacts)
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_07_padmm_solve_with_acceleration_and_container_warmstart.pdf"
            save_solver_info(solver=solver, path=str(path))

    def test_08_padmm_solve_single_contact(self):
        """
        Tests the Proximal-ADMM (PADMM) solver with default config on the reference problem (no
        constraints and limits) with a single contact.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=1, device=self.default_device)

        # Create the PADMM solver
        solver = PADMMSolver(model=test.model, collect_info=self.savefig)

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)

        # Extract solver info
        if self.savefig and solver.data.info is not None:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_08_padmm_solve.pdf"
            save_solver_info(solver=solver, path=str(path))

        # Check solution
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

    def test_09_apadmm_restart_matches_bilateral_reference(self):
        """Match a NumPy reference across an APADMM solve whose final iteration restarts.

        The restart tolerance is tight enough to reject every step, so this pins the
        combined-residual formula, the restart bookkeeping, and the rule that a restart
        rewinds only the extrapolation while keeping the current iterate.
        """
        test = TestSetup(
            builder_fn=basics.build_box_pendulum,
            max_world_contacts=1,
            device=self.default_device,
            ground=False,
        )
        config = PADMMSolver.Config(
            primal_tolerance=0.0,
            dual_tolerance=0.0,
            compl_tolerance=0.0,
            restart_tolerance=1.0e-6,
            max_iterations=6,
        )

        test.build()
        D = extract_delassus(test.problem.delassus, only_active_dims=True)[0]
        v_f = extract_problem_vector(
            test.problem.delassus,
            test.problem.data.v_f.numpy(),
            only_active_dims=True,
        )[0]
        P = extract_problem_vector(
            test.problem.delassus,
            test.problem.data.P.numpy(),
            only_active_dims=True,
        )[0]
        reference = solve_apadmm_bilateral_reference(D, v_f, P, config)

        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=True,
            use_graph_conditionals=False,
            collect_info=True,
        )
        solver.coldstart()
        solver.solve(problem=test.problem)

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status[11]), reference["restart"])
        self.assertEqual(int(status[12]), reference["num_restarts"])

        # The tight restart tolerance makes the solve alternate between accepted and
        # restarted steps, so both writeback branches run and the run ends on a restart.
        self.assertEqual(reference["restart"], 1)
        self.assertEqual(reference["num_restarts"], 3)

        iterations = int(status[1])
        reference_history = reference["history"]
        solver_history = {
            "a": solver.data.info.a.numpy()[:iterations],
            "norm_x": solver.data.info.norm_x.numpy()[:iterations],
            "norm_y": solver.data.info.norm_y.numpy()[:iterations],
            "norm_z": solver.data.info.norm_z.numpy()[:iterations],
            "r_dx": solver.data.info.r_dx.numpy()[:iterations],
            "r_dy": solver.data.info.r_dy.numpy()[:iterations],
            "r_dz": solver.data.info.r_dz.numpy()[:iterations],
            "r_comb": solver.data.info.r_comb.numpy()[:iterations],
            "r_comb_ratio": solver.data.info.r_comb_ratio.numpy()[:iterations],
            "num_restarts": solver.data.info.num_restarts.numpy()[:iterations],
        }
        for name, values in solver_history.items():
            np.testing.assert_allclose(values, reference_history[name], rtol=1e-5, atol=1e-6, err_msg=name)

        ncts = D.shape[0]
        np.testing.assert_allclose(solver.data.state.x_p.numpy()[:ncts], reference["x_p"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(solver.data.state.y_p.numpy()[:ncts], reference["y_p"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(solver.data.state.z_p.numpy()[:ncts], reference["z_p"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(solver.data.state.y_hat.numpy()[:ncts], reference["y_hat"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(solver.data.state.z_hat.numpy()[:ncts], reference["z_hat"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            solver.data.solution.lambdas.numpy()[:ncts],
            P * reference["y"],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            solver.data.solution.v_plus.numpy()[:ncts],
            reference["z"] / P,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_10_apadmm_converged_iteration_does_not_restart(self):
        """Suppress the acceleration restart on the iteration that converges.

        The restart tolerance is tight enough that every iteration past the first fails the
        combined-residual test, while the convergence tolerances are loose enough that the
        solve converges on that same iteration. The restart must then be suppressed, so that
        a solve reported as converged never also reports a restart it never acted on.
        """
        test = TestSetup(
            builder_fn=basics.build_box_pendulum,
            max_world_contacts=1,
            device=self.default_device,
            ground=False,
        )
        config = PADMMSolver.Config(
            primal_tolerance=1.0e3,
            dual_tolerance=1.0e3,
            compl_tolerance=1.0e3,
            restart_tolerance=1.0e-6,
            max_iterations=8,
        )

        test.build()
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=True,
            use_graph_conditionals=False,
        )
        solver.coldstart()
        solver.solve(problem=test.problem)

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status[0]), 1)
        # Convergence is gated on having completed more than one iteration.
        self.assertEqual(int(status[1]), 2)
        self.assertEqual(int(status[11]), 0)
        self.assertEqual(int(status[12]), 0)

    def test_11_apadmm_honors_per_world_iteration_budgets(self):
        """Give each world exactly its own iteration budget when the budgets differ.

        A world that exhausts its budget must stop advancing rather than keep stepping on
        the shared loop, and must release the shared completion counter only once, so that
        a world with a larger budget is not terminated early alongside it.
        """
        budgets = [3, 8]
        modes = [False, True] if self.default_device.is_cuda else [False]

        for use_graph_conditionals in modes:
            with self.subTest(use_graph_conditionals=use_graph_conditionals):
                test = TestSetup(
                    builder_fn=make_homogeneous_builder,
                    max_world_contacts=1,
                    device=self.default_device,
                    num_worlds=len(budgets),
                    build_fn=basics.build_box_pendulum,
                    ground=False,
                )
                configs = [
                    PADMMSolver.Config(
                        primal_tolerance=0.0,
                        dual_tolerance=0.0,
                        compl_tolerance=0.0,
                        max_iterations=budget,
                    )
                    for budget in budgets
                ]

                test.build()
                solver = PADMMSolver(
                    model=test.model,
                    config=configs,
                    warmstart=PADMMWarmStartMode.NONE,
                    use_acceleration=True,
                    use_graph_conditionals=use_graph_conditionals,
                )
                solver.coldstart()
                solver.solve(problem=test.problem)

                status = solver.data.status.numpy()
                self.assertEqual([int(status[w][1]) for w in range(len(budgets))], budgets)

    def assert_coulomb_cone_projection(self, cone: str, case: str, mu: float, x: list[float], expected: list[float]):
        """Assert one primal or dual Coulomb cone projection."""
        is_primal = cone == "primal"

        @wp.kernel
        def project(x: wp.array[wp.vec3f], mu: wp.array[wp.float32], y: wp.array[wp.vec3f]):
            if wp.static(is_primal):
                y[0] = project_to_coulomb_cone(x[0], mu[0])
            else:
                y[0] = project_to_coulomb_dual_cone(x[0], mu[0])

        x_arg = wp.array(np.array([x], dtype=np.float32), dtype=wp.vec3f, device=self.default_device)
        mu_arg = wp.array(np.array([mu], dtype=np.float32), dtype=wp.float32, device=self.default_device)
        y = wp.zeros(1, dtype=wp.vec3f, device=self.default_device)
        wp.launch(kernel=project, dim=1, inputs=[x_arg, mu_arg], outputs=[y], device=self.default_device)
        with self.subTest(mu=mu, cone=cone, case=case):
            np.testing.assert_allclose(y.numpy()[0], expected, rtol=1.0e-6, atol=1.0e-6)

    def test_12_coulomb_cone_projections_handle_zero_friction(self):
        """Handle the ray and half-space projections at zero friction."""
        mu = 0.0

        cone = "primal"
        self.assert_coulomb_cone_projection(cone, "positive ray", mu, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        self.assert_coulomb_cone_projection(cone, "polar", mu, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0])
        self.assert_coulomb_cone_projection(cone, "projection", mu, [1.0, 0.0, 1.0], [0.0, 0.0, 1.0])

        cone = "dual"
        self.assert_coulomb_cone_projection(cone, "pos. half-space", mu, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        self.assert_coulomb_cone_projection(cone, "polar", mu, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0])
        self.assert_coulomb_cone_projection(cone, "projection", mu, [1.0, 0.0, -1.0], [1.0, 0.0, 0.0])

    def test_13_coulomb_cone_projections_handle_all_branches(self):
        """Handle interior, polar, and general projections at positive friction."""
        for mu in (1.0e-6, 1.0, 1e6):
            den = 1.0 + mu * mu

            cone = "primal"
            self.assert_coulomb_cone_projection(cone, "interior", mu, [0.5 * mu, 0.0, 1.0], [0.5 * mu, 0.0, 1.0])
            self.assert_coulomb_cone_projection(cone, "polar", mu, [1.0, 0.0, -2.0 * mu], [0.0, 0.0, 0.0])
            self.assert_coulomb_cone_projection(cone, "projection", mu, [1.0, 0.0, 0.0], [mu * mu / den, 0.0, mu / den])

            cone = "dual"
            self.assert_coulomb_cone_projection(cone, "interior", mu, [1.0, 0.0, 2.0 * mu], [1.0, 0.0, 2.0 * mu])
            self.assert_coulomb_cone_projection(cone, "polar", mu, [1.0, 0.0, -2.0 / mu], [0.0, 0.0, 0.0])
            self.assert_coulomb_cone_projection(cone, "projection", mu, [1.0, 0.0, 0.0], [1.0 / den, 0.0, mu / den])


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
