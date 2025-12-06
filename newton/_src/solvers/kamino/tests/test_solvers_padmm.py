# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Proximal-ADMM Solver."""

import os
import unittest

import numpy as np
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.math import screw, vec3f
from newton._src.solvers.kamino.core.model import Model
from newton._src.solvers.kamino.dynamics.dual import DualProblem
from newton._src.solvers.kamino.kinematics.constraints import unpack_constraint_solutions
from newton._src.solvers.kamino.linalg import LLTBlockedSolver
from newton._src.solvers.kamino.linalg.utils.matrix import SquareSymmetricMatrixProperties
from newton._src.solvers.kamino.linalg.utils.range import in_range_via_gaussian_elimination
from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_box_pendulum,  # noqa: F401
    build_boxes_fourbar,  # noqa: F401
    build_boxes_hinged,  # noqa: F401
    build_boxes_nunchaku,  # noqa: F401
)
from newton._src.solvers.kamino.solvers.padmm import PADMMSettings, PADMMSolver, PADMMWarmStartMode
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_delassus,
    extract_info_vectors,
    extract_problem_vector,
)
from newton._src.solvers.kamino.tests.utils.make import make_containers, update_containers
from newton._src.solvers.kamino.utils import logger as msg

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
        device: Devicelike = None,
        **kwargs,
    ):
        # Cache the max contacts allocated for the test problem
        self.max_world_contacts = max_world_contacts

        # Construct the model description using model builders for different systems
        self.builder: ModelBuilder = builder_fn(**kwargs)

        # Set ad-hoc configurations
        self.builder.gravity[0].enabled = gravity
        if perturb:
            u_0 = screw(vec3f(+10.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0))
            for body in self.builder.bodies:
                body.u_i_0 = u_0

        # Create the model and containers from the builder
        self.model, self.data, self.limits, self.detector, self.jacobians = make_containers(
            builder=self.builder, max_world_contacts=max_world_contacts, device=device
        )
        self.contacts = self.detector.contacts
        self.state_p = self.model.state()

        # Create the DualProblem to be solved
        self.problem = DualProblem(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            solver=LLTBlockedSolver,
            device=device,
        )

        # Update the sim data containers
        update_containers(
            model=self.model, data=self.data, limits=self.limits, detector=self.detector, jacobians=self.jacobians
        )

    def build(self):
        # Build the dual problem
        self.problem.build(
            model=self.model,
            data=self.data,
            limits=self.limits.data,
            contacts=self.contacts.data,
            jacobians=self.jacobians.data,
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
    test: unittest.TestCase, model: Model, problem: DualProblem, solver: PADMMSolver, verbose: bool = False
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
        I_np = dtype.type(solver.settings[0].eta + solver.settings[0].rho_0) * np.eye(D_wp_np[w].shape[0], dtype=dtype)
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
        test.assertLessEqual(iterations, solver.settings[w].max_iterations)
        test.assertLessEqual(r_p, solver.settings[w].primal_tolerance)
        test.assertLessEqual(r_d, solver.settings[w].dual_tolerance)
        test.assertLessEqual(r_c, solver.settings[w].compl_tolerance)
        test.assertLessEqual(error_dual_abs_l2, solver.settings[w].dual_tolerance)
        test.assertLessEqual(error_dual_abs_inf, solver.settings[w].dual_tolerance)


def save_solver_info(solver: PADMMSolver, path: str | None = None, verbose: bool = False):
    # Attempt to import matplotlib for plotting
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
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
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for detailed output
        self.savefig = False  # Set to True to generate solver info plots
        self.output_dir = os.path.dirname(os.path.realpath(__file__)) + "/output/test_solvers_padmm"

        # Create output directory if saving figures
        if self.savefig:
            os.makedirs(self.output_dir, exist_ok=True)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_padmm_default(self):
        """
        Test creating a PADMMSolver with default initialization.
        """
        # Creating a default PADMMSolver without any model or settings
        # should result in a solver without any memory allocation.
        solver = PADMMSolver()
        self.assertIsNone(solver._size)
        self.assertIsNone(solver._data)
        self.assertIsNone(solver._device)
        self.assertEqual(solver.settings, [])

        # Requesting the solver data container when the
        # solver has not been finalized should raise an
        # error since no allocations have been made.
        self.assertRaises(RuntimeError, lambda: solver.data)

    def test_01_finalize_padmm_default(self):
        """
        Test creating a PADMMSolver with default initialization and then finalizing all memory allocations.
        """
        # Create a test setup
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Creating a default PADMMSolver without any model or settings
        # should result in a solver without any memory allocation.
        solver = PADMMSolver()

        # Finalize the solver with a model
        solver.finalize(test.model)

        # Check that the solver has been properly allocated
        self.assertIsNotNone(solver._size)
        self.assertIsNotNone(solver._data)
        self.assertIsNotNone(solver._device)
        self.assertEqual(len(solver.settings), test.model.size.num_worlds)
        self.assertIs(solver._device, test.model.device)
        self.assertIs(solver.size, test.model.size)

    def test_02_padmm_solve(self):
        """
        Tests the Proximal-ADMM (PADMM) solver with default settings on the reference problem.
        """
        # Create the test problem
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver settings
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.eta = 1e-5
        settings.rho_0 = 1.0
        settings.max_iterations = 200

        # Create the PADMM solver
        solver = PADMMSolver(
            model=test.model,
            settings=settings,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=False,
            collect_info=True,
        )

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_dir + "/test_02_padmm_solve.pdf"
            save_solver_info(solver=solver, path=path)

    def test_03_padmm_solve_with_acceleration(self):
        """
        Tests the Accelerated Proximal-ADMM (APADMM) solver on the reference problem with Nesterov acceleration.
        """
        # Create the test problem
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver settings
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.restart_tolerance = 0.999
        settings.eta = 1e-5
        settings.rho_0 = 1.0
        settings.max_iterations = 200

        # Create the PADMM solver
        solver = PADMMSolver(
            model=test.model,
            settings=settings,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=True,
            collect_info=True,
        )

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_dir + "/test_03_padmm_solve_with_acceleration.pdf"
            save_solver_info(solver=solver, path=path)

    def test_04_padmm_solve_with_internal_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with internal warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver settings
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.eta = 1e-5
        settings.rho_0 = 1.0
        settings.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            settings=settings,
            warmstart=PADMMWarmStartMode.INTERNAL,
            use_acceleration=False,
            collect_info=True,
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
            path = self.output_dir + "/test_04_padmm_solve_with_internal_warmstart.pdf"
            save_solver_info(solver=solver, path=path)

    def test_05_padmm_solve_with_container_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver settings
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.eta = 1e-5
        settings.rho_0 = 1.0
        settings.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            settings=settings,
            warmstart=PADMMWarmStartMode.CONTAINERS,
            use_acceleration=False,
            collect_info=True,
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
            path = self.output_dir + "/test_05_padmm_solve_with_container_warmstart.pdf"
            save_solver_info(solver=solver, path=path)

    def test_06_padmm_solve_with_acceleration_and_internal_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver settings
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.restart_tolerance = 0.999
        settings.eta = 1e-5
        settings.rho_0 = 1.0
        settings.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            settings=settings,
            warmstart=PADMMWarmStartMode.INTERNAL,
            use_acceleration=True,
            collect_info=True,
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
            path = self.output_dir + "/test_06_padmm_solve_with_acceleration_and_internal_warmstart.pdf"
            save_solver_info(solver=solver, path=path)

    def test_07_padmm_solve_with_acceleration_and_container_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver settings
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.restart_tolerance = 0.999
        settings.eta = 1e-5
        settings.rho_0 = 1.0
        settings.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            settings=settings,
            warmstart=PADMMWarmStartMode.CONTAINERS,
            use_acceleration=True,
            collect_info=True,
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
            path = self.output_dir + "/test_07_padmm_solve_with_acceleration_and_container_warmstart.pdf"
            save_solver_info(solver=solver, path=path)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=1000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
