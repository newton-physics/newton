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

"""Unit tests for the Proximal-ADMM Constraint Solver"""

import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.math import screw, vec3f
from newton._src.solvers.kamino.dynamics.dual import DualProblem
from newton._src.solvers.kamino.linalg import LLTBlockedSolver
from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_box_pendulum,  # noqa: F401
    build_boxes_fourbar,  # noqa: F401
    build_boxes_hinged,  # noqa: F401
    build_boxes_nunchaku,  # noqa: F401
)
from newton._src.solvers.kamino.models.utils import (
    make_single_builder,
)

# Module to be tested
from newton._src.solvers.kamino.solvers.padmm import PADMMSettings, PADMMSolver
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_delassus,
    extract_info_vectors,
    extract_problem_vector,
)

# Test utilities
from newton._src.solvers.kamino.tests.utils.make import make_containers, update_containers
from newton._src.solvers.kamino.tests.utils.print import print_model_info

###
# Helper functions
###


def print_problem_summary(problem: DualProblem):
    print("Dual Problem Summary:")
    print(f"problem.data.num_worlds: {problem.data.num_worlds}")
    print(f"problem.data.num_maxdims: {problem.data.num_maxdims}")
    print(f"problem.data.nl:{problem.data.nl}")
    print(f"problem.data.nc:{problem.data.nc}")
    print(f"problem.data.config: {problem.data.config}")
    print(f"problem.data.maxdim: {problem.data.maxdim}")
    print(f"problem.data.dim: {problem.data.dim}")
    print(f"problem.data.mio: {problem.data.mio}")
    print(f"problem.data.vio: {problem.data.vio}")
    print(f"problem.data.u_f (shape): {problem.data.u_f.shape}")
    print(f"problem.data.v_b (shape): {problem.data.v_b.shape}")
    print(f"problem.data.v_i (shape): {problem.data.v_i.shape}")
    print(f"problem.data.v_f (shape): {problem.data.v_f.shape}")
    print(f"problem.data.mu (shape): {problem.data.mu.shape}")
    print(f"problem.data.D (shape): {problem.data.D.shape}")


def print_solver_summary(solver: PADMMSolver):
    print("PADMM Solver Summary:")
    print("solver.size.num_worlds: ", solver.size.num_worlds)
    print("solver.size.max_limits: ", solver.size.sum_of_max_limits)
    print("solver.size.max_contacts: ", solver.size.sum_of_max_contacts)
    print("solver.size.max_unilaterals: ", solver.size.sum_of_max_unilaterals)
    print("solver.size.max_total_cts: ", solver.size.sum_of_max_total_cts)
    print("solver.size.max_iters: ", solver._max_iters)
    print("solver.data.config: ", solver.data.config)
    print("solver.data.status: ", solver.data.status)
    print("solver.data.penalty: ", solver.data.penalty)
    print("solver.data.info.offsets: ", solver.data.info.offsets)
    print("solver.data.info.r_primal: ", solver.data.info.r_primal.shape)
    print("solver.data.info.r_dual: ", solver.data.info.r_dual.shape)
    print("solver.data.info.r_compl: ", solver.data.info.r_compl.shape)
    print("solver.data.state.s:", solver.data.state.s.shape)
    print("solver.data.state.v:", solver.data.state.v.shape)
    print("solver.data.state.x:", solver.data.state.x.shape)
    print("solver.data.state.x_p:", solver.data.state.x_p.shape)
    print("solver.data.state.y:", solver.data.state.y.shape)
    print("solver.data.state.y_p:", solver.data.state.y_p.shape)
    print("solver.data.state.z:", solver.data.state.z.shape)
    print("solver.data.state.z_p:", solver.data.state.z_p.shape)
    print("solver.data.solution.v_plus:", solver.data.solution.v_plus.shape)
    print("solver.data.solution.lambdas:", solver.data.solution.lambdas.shape)


def save_solver_info(solver: PADMMSolver, path: str | None = None, verbose: bool = False):
    nw = solver.size.num_worlds
    status = solver.data.status.numpy()
    iterations = [status[w][1] for w in range(nw)]
    offsets_np = solver.data.info.offsets.numpy()
    num_restarts_np = extract_info_vectors(offsets_np, solver.data.info.num_restarts.numpy(), iterations)
    num_rho_updates_np = extract_info_vectors(offsets_np, solver.data.info.num_rho_updates.numpy(), iterations)
    a_np = extract_info_vectors(offsets_np, solver.data.info.a.numpy(), iterations)
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
    r_comb_np = extract_info_vectors(offsets_np, solver.data.info.r_comb.numpy(), iterations)
    r_comb_ratio_np = extract_info_vectors(offsets_np, solver.data.info.r_comb_ratio.numpy(), iterations)
    r_ncp_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_primal.numpy(), iterations)
    r_ncp_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_dual.numpy(), iterations)
    r_ncp_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_compl.numpy(), iterations)
    r_ncp_natmap_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_natmap.numpy(), iterations)

    if verbose:
        for w in range(nw):
            print(f"[World {w}] =======================================================================")
            print(f"solver.info.num_restarts: {num_restarts_np[w]}")
            print(f"solver.info.num_rho_updates: {num_rho_updates_np[w]}")
            print(f"solver.info.a: {a_np[w]}")
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
            print(f"solver.info.r_comb: {r_comb_np[w]}")
            print(f"solver.info.r_comb_ratio: {r_comb_ratio_np[w]}")
            print(f"solver.info.r_ncp_primal: {r_ncp_primal_np[w]}")
            print(f"solver.info.r_ncp_dual: {r_ncp_dual_np[w]}")
            print(f"solver.info.r_ncp_compl: {r_ncp_compl_np[w]}")
            print(f"solver.info.r_ncp_natmap: {r_ncp_natmap_np[w]}")

    # List of (label, data) for plotting
    info_list = [
        ("num_restarts", num_restarts_np),
        ("num_rho_updates", num_rho_updates_np),
        ("a", a_np),
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
        ("r_comb", r_comb_np),
        ("r_comb_ratio", r_comb_ratio_np),
        ("r_ncp_primal", r_ncp_primal_np),
        ("r_ncp_dual", r_ncp_dual_np),
        ("r_ncp_compl", r_ncp_compl_np),
        ("r_ncp_natmap", r_ncp_natmap_np),
    ]

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
        self.verbose = False  # Set to True for detailed output
        self.savefig = False  # Set to True to generate solver info plots
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_01_solve_box_on_plane_with_padmm(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the box-on-plane problem.
        """
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = make_single_builder(build_fn=build_box_on_plane)
        # builder = make_single_builder(build_fn=build_boxes_hinged, ground=True)
        # builder = make_single_builder(build_fn=build_boxes_nunchaku)
        # builder = make_single_builder(build_fn=build_boxes_fourbar)
        # builder = make_homogeneous_builder(num_worlds=4, build_fn=build_box_on_plane)
        # builder = make_homogeneous_builder(num_worlds=4, build_fn=build_boxes_hinged)
        # builder = make_homogeneous_builder(num_worlds=4, build_fn=build_boxes_nunchaku)
        # builder = make_homogeneous_builder(num_worlds=4, build_fn=build_boxes_fourbar)
        # builder = make_heterogeneous_builder()

        # Set ad-hoc configurations
        builder.gravity[0].enabled = True
        u_0 = screw(vec3f(+10.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0))
        for body in builder.bodies:
            body.u_i_0 = u_0

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)
        if self.verbose:
            print("\n")  # Print a newline for better readability
            print_model_info(model)
            print("\n")  # Print a newline for better readability

        # Create the Delassus operator
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            solver=LLTBlockedSolver,
            device=self.default_device,
        )

        # Build the dual problem
        problem.build(
            model=model, data=data, limits=limits.data, contacts=detector.contacts.data, jacobians=jacobians.data
        )

        # Optional verbose output
        if self.verbose:
            print("\n")  # Print a newline for better readability
            print_problem_summary(problem)
            print("\n")  # Print a newline for better readability

        # Define custom solver settings
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.restart_tolerance = 0.999
        settings.eta = 1e-5
        settings.rho_0 = 1.0  # 9.7  # 2.7
        settings.omega = 1.0  # 1.99
        settings.max_iterations = 500

        # Create the ADMM solver
        solver = PADMMSolver(
            model=model,
            limits=limits,
            contacts=detector.contacts,
            settings=settings,
            use_acceleration=False,
            collect_info=True,
            device=self.default_device,
        )

        # Solve the example problem
        solver.solve(problem=problem)

        # Extract numpy arrays from the solver state and solution
        only_active_dims = True
        D_wp_np = extract_delassus(problem.delassus, only_active_dims=only_active_dims)
        v_i_wp_np = extract_problem_vector(
            problem.delassus, problem.data.v_i.numpy(), only_active_dims=only_active_dims
        )
        v_b_wp_np = extract_problem_vector(
            problem.delassus, problem.data.v_b.numpy(), only_active_dims=only_active_dims
        )
        v_f_wp_np = extract_problem_vector(
            problem.delassus, problem.data.v_f.numpy(), only_active_dims=only_active_dims
        )
        P_wp_np = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=only_active_dims)
        s_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.s.numpy(), only_active_dims=only_active_dims
        )
        v_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.v.numpy(), only_active_dims=only_active_dims
        )
        x_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.x.numpy(), only_active_dims=only_active_dims
        )
        y_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.y.numpy(), only_active_dims=only_active_dims
        )
        z_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.z.numpy(), only_active_dims=only_active_dims
        )
        v_plus_wp_np = extract_problem_vector(
            problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=only_active_dims
        )
        lambdas_wp_np = extract_problem_vector(
            problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=only_active_dims
        )
        r_primal_wp_np = extract_problem_vector(
            problem.delassus, solver.data.residuals.r_primal.numpy(), only_active_dims=only_active_dims
        )
        r_dual_wp_np = extract_problem_vector(
            problem.delassus, solver.data.residuals.r_dual.numpy(), only_active_dims=only_active_dims
        )
        r_compl_wp_np = extract_problem_vector(
            problem.delassus, solver.data.residuals.r_compl.numpy(), only_active_dims=only_active_dims
        )
        v_plus_info_np = extract_problem_vector(
            problem.delassus, solver.data.info.v_plus.numpy(), only_active_dims=only_active_dims
        )
        v_aug_info_np = extract_problem_vector(
            problem.delassus, solver.data.info.v_aug.numpy(), only_active_dims=only_active_dims
        )
        s_info_np = extract_problem_vector(
            problem.delassus, solver.data.info.s.numpy(), only_active_dims=only_active_dims
        )

        # Optional verbose output
        status = solver.data.status.numpy()
        for w in range(model.size.num_worlds):
            # Recover the original (preconditioned) Delassua matrix from the in-place regularized storage
            dtype = D_wp_np[w].dtype
            I_np = dtype.type(settings.eta + settings.rho_0) * np.eye(D_wp_np[w].shape[0], dtype=dtype)
            D = D_wp_np[w] - I_np

            # Compute spectral properties of preconditioned D
            norm_v_f = np.linalg.norm(v_f_wp_np[w])
            eigvals = np.linalg.eigvalsh(D)
            eig_min = np.min(eigvals)
            eig_max = np.max(eigvals)
            L = eig_max + settings.eta
            m = max(eig_min, 0.0) + settings.eta
            kappa_D = L / m

            # Recover original Delassus matrix and v_f from preconditioned versions
            D_true = np.diag(np.reciprocal(P_wp_np[w])) @ D @ np.diag(np.reciprocal(P_wp_np[w]))
            v_f_true = np.diag(np.reciprocal(P_wp_np[w])) @ v_f_wp_np[w]

            # Compute the true dual solution and error
            v_plus_true = np.matmul(D_true, lambdas_wp_np[w]) + v_f_true
            error_dual_abs_l2 = np.linalg.norm(v_plus_true - v_plus_wp_np[w], ord=2)
            error_dual_abs_inf = np.linalg.norm(v_plus_true - v_plus_wp_np[w], ord=np.inf)

            # Check results
            self.assertEqual(status[w][0], 1)
            self.assertLessEqual(status[w][1], settings.max_iterations)
            self.assertLessEqual(status[w][2], settings.primal_tolerance)
            self.assertLessEqual(status[w][3], settings.dual_tolerance)
            self.assertLessEqual(status[w][4], settings.compl_tolerance)
            self.assertLess(error_dual_abs_l2, 1e-5)
            self.assertLess(error_dual_abs_inf, 1e-5)

            # Optionally print relevant solver data
            if self.verbose:
                print("\n")  # Print a newline for better readability
                print(f"[World {w}] =======================================================================")
                print(f"D_reg:\n{D_wp_np[w]}")
                print(f"D:\n{D}")
                print(f"v_i:\n{v_i_wp_np[w]}")
                print(f"v_b:\n{v_b_wp_np[w]}")
                print(f"v_f:\n{v_f_wp_np[w]}")
                print(f"P:\n{P_wp_np[w]}")
                print(f"norm_v_f: {norm_v_f}")
                print(f"eig_max: {eig_max}")
                print(f"eig_min: {eig_min}")
                print(f"L: {L}")
                print(f"m: {m}")
                print(f"kappa_D: {kappa_D}")
                print("---------")
                print(f"s: {s_wp_np[w]}")
                print(f"v: {v_wp_np[w]}")
                print(f"x: {x_wp_np[w]}")
                print(f"y: {y_wp_np[w]}")
                print(f"z: {z_wp_np[w]}")
                print("---------")
                print(f" v_plus: {v_plus_wp_np[w]}")
                print(f"lambdas: {lambdas_wp_np[w]}")
                print("---------")
                print(f"r_primal: {r_primal_wp_np[w]}")
                print(f"  r_dual: {r_dual_wp_np[w]}")
                print(f" r_compl: {r_compl_wp_np[w]}")
                print("---------")
                print(f"iterations: {status[w][1]}")
                print(f"converged: {status[w][0]}")
                print(f"r_p: {status[w][2]}")
                print(f"r_d: {status[w][3]}")
                print(f"r_c: {status[w][4]}")
                print("---------")
                print(f"error_dual_abs_l2: {error_dual_abs_l2}")
                print(f"error_dual_abs_inf: {error_dual_abs_inf}")
                print("---------")
                print(f"   state:      s: {s_wp_np[w]}")
                print(f"    info:      s: {s_info_np[w]}")
                print(f"   state:      y: {y_wp_np[w]}")
                print(f"solution: lambda: {lambdas_wp_np[w]}")
                print(f"   state:      z: {z_wp_np[w]}")
                print(f"    info:  v_aug: {v_aug_info_np[w]}")
                print(f"solution: v_plus: {v_plus_wp_np[w]}")
                print(f"    info: v_plus: {v_plus_info_np[w]}")
                print(f"    true: v_plus: {v_plus_true}")
                print("\n\n")  # Print a newline for better readability

        # Extract solver info
        if self.savefig:
            print("Generating solver info plots...")
            path = os.path.dirname(os.path.realpath(__file__)) + "/output/test_solver_padmm_solver_info_01.pdf"
            save_solver_info(solver=solver, path=path)

    def test_02_solve_box_on_plane_with_accelerated_padmm(self):
        """
        Tests the Accelerated Proximal-ADMM (APADMM) solver on the box-on-plane problem.
        """
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = make_single_builder(build_fn=build_box_on_plane)
        # builder = make_single_builder(build_fn=build_boxes_hinged, ground=True)
        # builder = make_single_builder(build_fn=build_boxes_nunchaku)
        # builder = make_single_builder(build_fn=build_boxes_fourbar)
        # builder = make_homogeneous_builder(num_worlds=4, build_fn=build_box_on_plane)
        # builder = make_homogeneous_builder(num_worlds=4, build_fn=build_boxes_hinged)
        # builder = make_homogeneous_builder(num_worlds=4, build_fn=build_boxes_nunchaku)
        # builder = make_homogeneous_builder(num_worlds=4, build_fn=build_boxes_fourbar)
        # builder = make_heterogeneous_builder()

        # Set ad-hoc configurations
        builder.gravity[0].enabled = True
        u_0 = screw(vec3f(+10.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0))
        for body in builder.bodies:
            body.u_i_0 = u_0

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)
        if self.verbose:
            print("\n")  # Print a newline for better readability
            print_model_info(model)
            print("\n")  # Print a newline for better readability

        # Create the Delassus operator
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            solver=LLTBlockedSolver,
            device=self.default_device,
        )

        # Build the dual problem
        problem.build(
            model=model, data=data, limits=limits.data, contacts=detector.contacts.data, jacobians=jacobians.data
        )

        # Optional verbose output
        if self.verbose:
            print("\n")  # Print a newline for better readability
            print_problem_summary(problem)
            print("\n")  # Print a newline for better readability

        # Define custom solver settings
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.restart_tolerance = 0.999
        settings.eta = 1e-5
        settings.rho_0 = 1.0  # 9.7  # 2.7
        settings.omega = 1.0  # 1.99
        settings.max_iterations = 500

        # Create the ADMM solver
        solver = PADMMSolver(
            model=model,
            limits=limits,
            contacts=detector.contacts,
            settings=settings,
            use_acceleration=True,
            collect_info=True,
            device=self.default_device,
        )

        # Solve the example problem
        solver.solve(problem=problem)

        # Extract numpy arrays from the solver state and solution
        only_active_dims = True
        D_wp_np = extract_delassus(problem.delassus, only_active_dims=only_active_dims)
        v_i_wp_np = extract_problem_vector(
            problem.delassus, problem.data.v_i.numpy(), only_active_dims=only_active_dims
        )
        v_b_wp_np = extract_problem_vector(
            problem.delassus, problem.data.v_b.numpy(), only_active_dims=only_active_dims
        )
        v_f_wp_np = extract_problem_vector(
            problem.delassus, problem.data.v_f.numpy(), only_active_dims=only_active_dims
        )
        P_wp_np = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=only_active_dims)
        s_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.s.numpy(), only_active_dims=only_active_dims
        )
        v_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.v.numpy(), only_active_dims=only_active_dims
        )
        x_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.x.numpy(), only_active_dims=only_active_dims
        )
        y_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.y.numpy(), only_active_dims=only_active_dims
        )
        z_wp_np = extract_problem_vector(
            problem.delassus, solver.data.state.z.numpy(), only_active_dims=only_active_dims
        )
        v_plus_wp_np = extract_problem_vector(
            problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=only_active_dims
        )
        lambdas_wp_np = extract_problem_vector(
            problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=only_active_dims
        )
        r_primal_wp_np = extract_problem_vector(
            problem.delassus, solver.data.residuals.r_primal.numpy(), only_active_dims=only_active_dims
        )
        r_dual_wp_np = extract_problem_vector(
            problem.delassus, solver.data.residuals.r_dual.numpy(), only_active_dims=only_active_dims
        )
        r_compl_wp_np = extract_problem_vector(
            problem.delassus, solver.data.residuals.r_compl.numpy(), only_active_dims=only_active_dims
        )
        v_plus_info_np = extract_problem_vector(
            problem.delassus, solver.data.info.v_plus.numpy(), only_active_dims=only_active_dims
        )
        v_aug_info_np = extract_problem_vector(
            problem.delassus, solver.data.info.v_aug.numpy(), only_active_dims=only_active_dims
        )
        s_info_np = extract_problem_vector(
            problem.delassus, solver.data.info.s.numpy(), only_active_dims=only_active_dims
        )

        # Optional verbose output
        status = solver.data.status.numpy()
        for w in range(model.size.num_worlds):
            # Recover the original (preconditioned) Delassua matrix from the in-place regularized storage
            dtype = D_wp_np[w].dtype
            I_np = dtype.type(settings.eta + settings.rho_0) * np.eye(D_wp_np[w].shape[0], dtype=dtype)
            D = D_wp_np[w] - I_np

            # Compute spectral properties of preconditioned D
            norm_v_f = np.linalg.norm(v_f_wp_np[w])
            eigvals = np.linalg.eigvalsh(D)
            eig_min = np.min(eigvals)
            eig_max = np.max(eigvals)
            L = eig_max + settings.eta
            m = max(eig_min, 0.0) + settings.eta
            kappa_D = L / m

            # Recover original Delassus matrix and v_f from preconditioned versions
            D_true = np.diag(np.reciprocal(P_wp_np[w])) @ D @ np.diag(np.reciprocal(P_wp_np[w]))
            v_f_true = np.diag(np.reciprocal(P_wp_np[w])) @ v_f_wp_np[w]

            # Compute the true dual solution and error
            v_plus_true = np.matmul(D_true, lambdas_wp_np[w]) + v_f_true
            error_dual_abs_l2 = np.linalg.norm(v_plus_true - v_plus_wp_np[w], ord=2)
            error_dual_abs_inf = np.linalg.norm(v_plus_true - v_plus_wp_np[w], ord=np.inf)

            # Check results
            self.assertEqual(status[w][0], 1)
            self.assertLessEqual(status[w][1], settings.max_iterations)
            self.assertLessEqual(status[w][2], settings.primal_tolerance)
            self.assertLessEqual(status[w][3], settings.dual_tolerance)
            self.assertLessEqual(status[w][4], settings.compl_tolerance)
            self.assertLess(error_dual_abs_l2, 1e-5)
            self.assertLess(error_dual_abs_inf, 1e-5)

            # Optionally print relevant solver data
            if self.verbose:
                print("\n")  # Print a newline for better readability
                print(f"[World {w}] =======================================================================")
                print(f"D_reg:\n{D_wp_np[w]}")
                print(f"D:\n{D}")
                print(f"v_i:\n{v_i_wp_np[w]}")
                print(f"v_b:\n{v_b_wp_np[w]}")
                print(f"v_f:\n{v_f_wp_np[w]}")
                print(f"P:\n{P_wp_np[w]}")
                print(f"norm_v_f: {norm_v_f}")
                print(f"eig_max: {eig_max}")
                print(f"eig_min: {eig_min}")
                print(f"L: {L}")
                print(f"m: {m}")
                print(f"kappa_D: {kappa_D}")
                print("---------")
                print(f"s: {s_wp_np[w]}")
                print(f"v: {v_wp_np[w]}")
                print(f"x: {x_wp_np[w]}")
                print(f"y: {y_wp_np[w]}")
                print(f"z: {z_wp_np[w]}")
                print("---------")
                print(f" v_plus: {v_plus_wp_np[w]}")
                print(f"lambdas: {lambdas_wp_np[w]}")
                print("---------")
                print(f"r_primal: {r_primal_wp_np[w]}")
                print(f"  r_dual: {r_dual_wp_np[w]}")
                print(f" r_compl: {r_compl_wp_np[w]}")
                print("---------")
                print(f"iterations: {status[w][1]}")
                print(f"converged: {status[w][0]}")
                print(f"r_p: {status[w][2]}")
                print(f"r_d: {status[w][3]}")
                print(f"r_c: {status[w][4]}")
                print("---------")
                print(f"error_dual_abs_l2: {error_dual_abs_l2}")
                print(f"error_dual_abs_inf: {error_dual_abs_inf}")
                print("---------")
                print(f"   state:      s: {s_wp_np[w]}")
                print(f"    info:      s: {s_info_np[w]}")
                print(f"   state:      y: {y_wp_np[w]}")
                print(f"solution: lambda: {lambdas_wp_np[w]}")
                print(f"   state:      z: {z_wp_np[w]}")
                print(f"    info:  v_aug: {v_aug_info_np[w]}")
                print(f"solution: v_plus: {v_plus_wp_np[w]}")
                print(f"    info: v_plus: {v_plus_info_np[w]}")
                print(f"    true: v_plus: {v_plus_true}")
                print("\n\n")  # Print a newline for better readability

        # Extract solver info
        if self.savefig:
            print("Generating solver info plots...")
            path = os.path.dirname(os.path.realpath(__file__)) + "/output/test_solver_padmm_solver_info_02.pdf"
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
