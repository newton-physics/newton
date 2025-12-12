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

"""
Defines the Proximal-ADMM solver class.

This is the highest-level interface for the Proximal-ADMM solver for constrained rigid multi-body systems.

See the :mod:`newton._src.solvers.kamino.solvers.padmm` module for a detailed description and usage example.
"""

import warp as wp
from warp.context import Devicelike

from ...core.model import Model, ModelData, ModelSize
from ...dynamics.dual import DualProblem
from ...geometry.contacts import Contacts
from ...kinematics.limits import Limits
from .kernels import (
    _apply_dual_preconditioner_to_solution,
    _apply_dual_preconditioner_to_state,
    _compute_complementarity_residuals,
    _compute_desaxce_correction,
    _compute_final_desaxce_correction,
    _compute_infnorm_residuals_serially,
    _compute_infnorm_residuals_serially_accel,
    _compute_projection_argument,
    _compute_solution_vectors,
    _compute_velocity_bias,
    _project_to_feasible_cone,
    _update_delassus_proximal_regularization,
    _update_state_with_acceleration,
    _warmstart_contact_constraints,
    _warmstart_desaxce_correction,
    _warmstart_joint_constraints,
    _warmstart_limit_constraints,
    make_collect_solver_info_kernel,
    make_initialize_solver_kernel,
    make_update_dual_variables_and_compute_primal_dual_residuals,
)
from .types import PADMMConfig, PADMMData, PADMMSettings, PADMMWarmStartMode

###
# Module interface
###

__all__ = [
    "PADMMSolver",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Interfaces
###


class PADMMSolver:
    """
    The Proximal-ADMM (PADMM) forward dynamics solver for constrained rigid multi-body systems.

    This solver implements the Proximal-ADMM algorithm to solve the Lagrange dual of the
    constrained forward dynamics problem in constraint reactions (i.e. impulses) and
    post-event constraint-space velocities.

    Notes:
    - is designed to work with the DualProblem formulation.
    - operates on the Lagrange dual of the constrained forward dynamics problem.
    - is based on the Proximal-ADMM algorithm, which introduces a proximal regularization term
    - supports multiple penalty update methods, including fixed, linear, and spectral updates.
    - can be configured with various tolerances, penalty parameters, and other settings.
    """

    def __init__(
        self,
        model: Model | None = None,
        settings: list[PADMMSettings] | PADMMSettings | None = None,
        warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
        use_acceleration: bool = True,
        collect_info: bool = False,
        device: Devicelike = None,
    ):
        """
        Initializes a PADMM solver.

        If a model is provided, it will perform all necessary memory allocations on the
        target device, otherwise the user must call `finalize()` before using the solver.

        Args:
            model (Model | None): The model for which to allocate the solver data.
            limits (Limits | None): The limits container associated with the model.
            contacts (Contacts | None): The contacts container associated with the model.
            settings (list[PADMMSettings] | PADMMSettings | None): The solver settings to use.
            use_acceleration (bool): Set to `True` to enable Nesterov acceleration.
            collect_info (bool): Set to `True` to enable collection of solver convergence info.\n
                This setting is intended only for analysis and debugging purposes, as it
                will increase memory consumption and reduce wall-clock time.
            device (Devicelike | None): The target device on which to allocate the solver data.
        """

        # Declare the internal solver settings cache
        self._warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE
        self._settings: list[PADMMSettings] = []
        self._use_acceleration: bool = False
        self._collect_info: bool = False

        # Declare the model size cache
        self._size: ModelSize | None = None

        # Declare the solver data container
        self._data: PADMMData | None = None

        # Declare the device cache
        self._device: Devicelike = None

        # Perform memory allocations if a model is provided
        if model is not None:
            self.finalize(
                model=model,
                settings=settings,
                warmstart=warmstart,
                use_acceleration=use_acceleration,
                collect_info=collect_info,
                device=device,
            )

    ###
    # Properties
    ###

    @property
    def settings(self) -> list[PADMMSettings]:
        """
        Returns the host-side cache of the solver settings.\n
        They are used to construct the warp array of type :class:`PADMMConfig` on the target device.
        """
        return self._settings

    @property
    def size(self) -> ModelSize:
        """
        Returns the host-side cache of the solver allocation sizes.
        """
        return self._size

    @property
    def data(self) -> PADMMData:
        """
        Returns a reference to the high-level solver data container.
        """
        if self._data is None:
            raise RuntimeError("Solver data has not been allocated yet. Call `finalize()` first.")
        return self._data

    @property
    def device(self) -> Devicelike:
        """
        Returns the device on which the solver data is allocated.
        """
        return self._device

    ###
    # Public API
    ###

    def finalize(
        self,
        model: Model | None = None,
        settings: list[PADMMSettings] | PADMMSettings | None = None,
        warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
        use_acceleration: bool = True,
        collect_info: bool = False,
        device: Devicelike = None,
    ):
        """
        Allocates the solver data structures on the specified device.

        Args:
            model (Model | None): The model for which to allocate the solver data.
            limits (Limits | None): The limits container associated with the model.
            contacts (Contacts | None): The contacts container associated with the model.
            settings (list[PADMMSettings] | PADMMSettings | None): The solver settings to use.
            use_acceleration (bool): Set to `True` to enable Nesterov acceleration.
            collect_info (bool): Set to `True` to enable collection of solver convergence info.\n
                This setting is intended only for analysis and debugging purposes, as it
                will increase memory consumption and reduce wall-clock time.
            device (Devicelike | None): The target device on which to allocate the solver data.
        """

        # Ensure the model is valid
        if model is None:
            raise ValueError("A model of type `Model` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, Model):
            raise ValueError("Invalid model provided. Must be an instance of `Model`.")

        # Cache a reference to the model size meta-data container
        self._size = model.size

        # Cache high-level solver options
        self._warmstart = warmstart
        self._use_acceleration = use_acceleration
        self._collect_info = collect_info

        # Set the target device if specified, otherwise use the model device
        self._device = device if device is not None else model.device

        # Cache the solver settings
        if settings is not None:
            self._settings = self._check_settings(model, settings)
        elif len(self._settings) == 0:
            self._settings = self._check_settings(model, None)

        # Compute the largest max iterations across all worlds
        # NOTE: This is needed to allocate the solver
        # info arrays if `collect_info` is enabled
        max_of_max_iters = max([s.max_iterations for s in self._settings])

        # Allocate memory in device global memory
        self._data = PADMMData(
            size=self._size,
            max_iters=max_of_max_iters,
            use_acceleration=self._use_acceleration,
            collect_info=self._collect_info,
            device=self._device,
        )

        # Write algorithm configs into device memory
        configs = [s.to_config() for s in self._settings]
        with wp.ScopedDevice(self._device):
            self._data.config = wp.array(configs, dtype=PADMMConfig)

        # Specialize certain solver kernels depending on whether acceleration is enabled
        self._initialize_solver_kernel = make_initialize_solver_kernel(use_acceleration)
        self._collect_solver_info_kernel = make_collect_solver_info_kernel(use_acceleration)
        self._update_dual_variables_and_compute_primal_dual_residuals_kernel = (
            make_update_dual_variables_and_compute_primal_dual_residuals(use_acceleration)
        )

    def reset(self):
        """
        Resets the all internal solver data to sentinel values.
        """
        # Initialize state arrays to zero
        self._data.state.reset(use_acceleration=self._use_acceleration)

        # Initialize solution arrays to zero
        self._data.solution.zero()

        # Reset solver info to zero if collection is enabled
        if self._collect_info:
            self._data.info.zero()

        # Reset the solver status and ALM penalty parameter
        self._initialize()

    def coldstart(self):
        """
        Initializes the internal solver state to perform a cold-start solve.\n
        This method sets all solver state variables to zeros.
        """
        # Initialize state arrays to zero
        self._data.state.reset(use_acceleration=self._use_acceleration)

    def warmstart(
        self,
        problem: DualProblem,
        model: Model,
        data: ModelData,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
    ):
        """
        Warm-starts the internal solver state based on the selected warm-start mode.

        Supported warm-start modes:
        - `PADMMWarmStartMode.NONE`: No warm-starting is performed.
        - `PADMMWarmStartMode.INTERNAL`: Warm-starts from the internal solution state.
        - `PADMMWarmStartMode.CONTAINERS`: Warm-starts from the provided limits and contacts containers.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.\n
                This is needed during warm-starts in order to access the problem preconditioning.
            model (Model): The model associated with the problem.
            data (ModelData): The model data associated with the problem.
            limits (Limits | None): The limits container associated with the model.\n
                If `None`, no warm-starting from limits is performed.
            contacts (Contacts | None): The contacts container associated with the model.\n
                If `None`, no warm-starting from contacts is performed.
        """
        # TODO: IS THIS EVEN NECESSARY AT ALL?
        # First reset the internal solver state to ensure proper initialization
        self._data.state.reset(use_acceleration=self._use_acceleration)

        # Warm-start based on the selected mode
        match self._warmstart:
            case PADMMWarmStartMode.NONE:
                return
            case PADMMWarmStartMode.INTERNAL:
                self._warmstart_from_solution(problem)
            case PADMMWarmStartMode.CONTAINERS:
                self._warmstart_from_containers(problem, model, data, limits, contacts)
            case _:
                raise ValueError(f"Invalid warmstart mode: {self._warmstart}")

    def solve(self, problem: DualProblem):
        """
        Solves the given dual problem using PADMM.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Initialize the solver status and ALM penalty parameter
        self._initialize()

        # Add the diagonal proximal regularization to the Delassus matrix
        # D_{eta,rho} := D + (eta + rho) * I_{ncts}
        self._update_regularization(problem)

        # Reset the solver info to zero if collection is enabled
        if self._collect_info:
            self._data.info.zero()

        # Iterate until convergence or maximum number of iterations is reached
        if self._use_acceleration:
            wp.capture_while(self._data.state.done, while_body=self._step_accel, problem=problem)
        else:
            wp.capture_while(self._data.state.done, while_body=self._step, problem=problem)

        # Update the final solution from the terminal PADMM state
        self._update_solution(problem)

    ###
    # Internals - High-Level Operations
    ###

    @staticmethod
    def _check_settings(
        model: Model | None = None, settings: list[PADMMSettings] | PADMMSettings | None = None
    ) -> list[PADMMSettings]:
        """
        Checks and validates the provided solver settings, returning a list
        of settings objects corresponding to each world in the model.

        Args:
            model (Model | None): The model for which to validate the settings.
            settings (list[PADMMSettings] | PADMMSettings | None): The solver settings to validate.
        """
        # If no settings are provided, use defaults
        if settings is None:
            # If no model is provided, use a single default settings object
            if model is None:
                settings = [PADMMSettings()]

            # If a model is provided, create a list of default settings
            # objects based on the number of worlds in the model
            else:
                num_worlds = model.info.num_worlds
                settings = [PADMMSettings()] * num_worlds

        # If a single settings object is provided, convert it to a list
        elif isinstance(settings, PADMMSettings):
            settings = [settings] * (model.info.num_worlds if model else 1)

        # If a list of settings is provided, ensure it matches the number
        # of worlds and that all settings are instances of PADMMSettings
        elif isinstance(settings, list):
            if model is not None and len(settings) != model.info.num_worlds:
                raise ValueError(f"Expected {model.info.num_worlds} settings, got {len(settings)}")
            if not all(isinstance(s, PADMMSettings) for s in settings):
                raise TypeError("All settings must be instances of PADMMSettings")

        # Return the validated settings
        return settings

    def _initialize(self):
        """
        Launches a kernel to initialize the internal solver state before starting a new solve.\n
        The kernel is parallelized over the number of worlds.
        """
        # Initialize solver status and penalty parameters from the set configurations
        wp.launch(
            kernel=self._initialize_solver_kernel,
            dim=self._size.num_worlds,
            inputs=[
                self._data.config,
                self._data.status,
                self._data.penalty,
                self._data.state.sigma,
                self._data.state.a_p,
            ],
        )

        # Initialize the global while condition flag
        # NOTE: We use a single-element array that is initialized
        # to number of worlds and decremented by each world that
        # converges or reaches the maximum number of iterations
        self._data.state.done.fill_(self._size.num_worlds)

    def _update_regularization(self, problem: DualProblem):
        """
        Updates the diagonal regularization of the lhs matrix with the proximal regularization terms.\n
        For `DualProblem` solves, the lhs matrix corresponds to the Delassus matrix.\n
        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Update the proximal regularization term in the Delassus matrix
        wp.launch(
            kernel=_update_delassus_proximal_regularization,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.mio,
                self._data.status,
                self._data.state.sigma,
                # Outputs:
                problem.data.D,
            ],
        )

        # Compute Choleky/LDLT factorization of the Delassus matrix
        problem._delassus.compute(reset_to_zero=True)

    def _step(self, problem: DualProblem):
        """
        Performs a single PADMM solver iteration.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute De Saxce correction from the previous dual variables
        self._update_desaxce_correction(problem, self._data.state.z_p)

        # Compute the total velocity bias, i.e. rhs vector of the unconstrained linear system
        self._update_velocity_bias(problem, self._data.state.y_p, self._data.state.z_p)

        # Compute the unconstrained solution and store in the primal variables
        self._update_unconstrained_solution(problem)

        # Compute the argument to the projection operator with over-relaxation
        self._update_projection_argument(problem, self._data.state.z_p)

        # Project the over-relaxed primal variables to the feasible set
        self._update_projection_to_feasible_set(problem)

        # Update the dual variables and compute residuals from the current state
        self._update_dual_variables_and_residuals(problem)

        # Compute infinity-norm of all residuals and check for convergence
        self._update_convergence_check(problem)

        # Optionally record internal solver info
        if self._collect_info:
            self._update_solver_info(problem)

        # Update caches of previous state variables
        self._update_previous_state()

    def _step_accel(self, problem: DualProblem):
        """
        Performs a single PADMM solver iteration with Nesterov acceleration.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute De Saxce correction from the previous dual variables
        self._update_desaxce_correction(problem, self._data.state.z_hat)

        # Compute the total velocity bias, i.e. rhs vector of the unconstrained linear system
        self._update_velocity_bias(problem, self._data.state.y_hat, self._data.state.z_hat)

        # Compute the unconstrained solution and store in the primal variables
        self._update_unconstrained_solution(problem)

        # Project the over-relaxed primal variables to the feasible set
        self._update_projection_argument(problem, self._data.state.z_hat)

        # Project the over-relaxed primal variables to the feasible set
        self._update_projection_to_feasible_set(problem)

        # Update the dual variables and compute residuals from the current state
        self._update_dual_variables_and_residuals_accel(problem)

        # Compute infinity-norm of all residuals and check for convergence
        self._update_convergence_check_accel(problem)

        # Optionally update Nesterov acceleration states from the current iteration
        self._update_acceleration(problem)

        # Optionally record internal solver info
        if self._collect_info:
            self._update_solver_info(problem)

        # Update caches of previous state variables
        self._update_previous_state_accel()

    ###
    # Internals - Warm-starting
    ###

    def _warmstart_desaxce_correction(self, problem: DualProblem, z: wp.array):
        """
        Applies the De Saxce correction to the provided post-event constraint-space velocity warm-start.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.\n
                This is needed during warm-starts in order to access the problem preconditioning.
            z (wp.array): The post-event constraint-space velocity warm-start variable.\n
                This can either be `z_p` or `z_hat` depending on whether acceleration is used.
        """
        wp.launch(
            kernel=_warmstart_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                # Outputs:
                z,
            ],
        )

    def _warmstart_joint_constraints(
        self,
        model: Model,
        data: ModelData,
        problem: DualProblem,
        x_0: wp.array,
        y_0: wp.array,
        z_0: wp.array,
    ):
        """
        Warm-starts the bilateral joint constraint variables from the model data container.

        Args:
            model (Model): The model associated with the problem.
            data (ModelData): The model data associated with the problem.
            problem (DualProblem): The dual forward dynamics problem to be solved.\n
                This is needed during warm-starts in order to access the problem preconditioning.
            x_0 (wp.array): The output primal variables array to be warm-started.
            y_0 (wp.array): The output slack variables array to be warm-started.
            z_0 (wp.array): The output dual variables array to be warm-started.
        """
        wp.launch(
            kernel=_warmstart_joint_constraints,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.time.dt,
                model.info.total_cts_offset,
                model.info.joint_cts_offset,
                model.joints.wid,
                model.joints.num_cts,
                model.joints.cts_offset,
                data.joints.lambda_j,
                problem.data.P,
                # Outputs:
                x_0,
                y_0,
                z_0,
            ],
        )

    def _warmstart_limit_constraints(
        self,
        model: Model,
        data: ModelData,
        limits: Limits,
        problem: DualProblem,
        x_0: wp.array,
        y_0: wp.array,
        z_0: wp.array,
    ):
        """
        Warm-starts the unilateral limit constraint variables from the limits data container.

        Args:
            model (Model): The model associated with the problem.
            data (ModelData): The model data associated with the problem.
            limits (Limits): The limits container associated with the model.
            problem (DualProblem): The dual forward dynamics problem to be solved.\n
                This is needed during warm-starts in order to access the problem preconditioning.
            x_0 (wp.array): The output primal variables array to be warm-started.
            y_0 (wp.array): The output slack variables array to be warm-started.
            z_0 (wp.array): The output dual variables array to be warm-started.
        """
        wp.launch(
            kernel=_warmstart_limit_constraints,
            dim=limits.model_max_limits_host,
            inputs=[
                # Inputs:
                model.time.dt,
                model.info.total_cts_offset,
                data.info.limit_cts_group_offset,
                limits.model_active_limits,
                limits.wid,
                limits.lid,
                limits.reaction,
                limits.velocity,
                problem.data.P,
                # Outputs:
                x_0,
                y_0,
                z_0,
            ],
        )

    def _warmstart_contact_constraints(
        self,
        model: Model,
        data: ModelData,
        contacts: Contacts,
        problem: DualProblem,
        x_0: wp.array,
        y_0: wp.array,
        z_0: wp.array,
    ):
        """
        Warm-starts the unilateral contact constraint variables from the contacts data container.

        Args:
            model (Model): The model associated with the problem.
            data (ModelData): The model data associated with the problem.
            contacts (Contacts): The contacts container associated with the model.
            problem (DualProblem): The dual forward dynamics problem to be solved.\n
                This is needed during warm-starts in order to access the problem preconditioning.
            x_0 (wp.array): The output primal variables array to be warm-started.
            y_0 (wp.array): The output slack variables array to be warm-started.
            z_0 (wp.array): The output dual variables array to be warm-started.
        """
        wp.launch(
            kernel=_warmstart_contact_constraints,
            dim=contacts.model_max_contacts_host,
            inputs=[
                # Inputs:
                model.time.dt,
                model.info.total_cts_offset,
                data.info.contact_cts_group_offset,
                contacts.model_active_contacts,
                contacts.wid,
                contacts.cid,
                contacts.material,
                contacts.reaction,
                contacts.velocity,
                problem.data.P,
                # Outputs:
                x_0,
                y_0,
                z_0,
            ],
        )

    def _warmstart_from_solution(self, problem: DualProblem):
        """
        Warm-starts the internal solver state from the stored solution variables.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.\n
                This is needed during warm-starts in order to access the problem preconditioning.
        """
        # Apply the dual-problem preconditioner to the stored solution
        # in order to project to solution space of the PADMM variables
        wp.launch(
            kernel=_apply_dual_preconditioner_to_solution,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                # Outputs:
                self._data.solution.lambdas,
                self._data.solution.v_plus,
            ],
        )

        # Capture references to the warm-start variables
        # depending on whether acceleration is used or not
        if self._use_acceleration:
            y_0 = self._data.state.y_hat
            z_0 = self._data.state.z_hat
        else:
            y_0 = self._data.state.y_p
            z_0 = self._data.state.z_p

        # Copy the last solution into the warm-start variables
        wp.copy(self._data.state.x_p, self._data.solution.lambdas)
        wp.copy(y_0, self._data.solution.lambdas)
        wp.copy(z_0, self._data.solution.v_plus)
        self._warmstart_desaxce_correction(problem, z=z_0)

    def _warmstart_from_containers(
        self,
        problem: DualProblem,
        model: Model,
        data: ModelData,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
    ):
        """
        Warm-starts the internal solver state from the provided model data and limits and contacts containers.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.\n
                This is needed during warm-starts in order to access the problem preconditioning.
            model (Model): The model associated with the problem.
            data (ModelData): The model data associated with the problem.
            limits (Limits | None): The limits container associated with the model.\n
                If `None`, no warm-starting from limits is performed.
            contacts (Contacts | None): The contacts container associated with the model.\n
                If `None`, no warm-starting from contacts is performed.
        """
        # Capture references to the warm-start variables
        # depending on whether acceleration is used or not
        x_0 = self._data.state.x_p
        if self._use_acceleration:
            y_0 = self._data.state.y_hat
            z_0 = self._data.state.z_hat
        else:
            y_0 = self._data.state.y_p
            z_0 = self._data.state.z_p

        # Warm-start each constraint group from constraint states cached in the data containers
        if model.size.sum_of_num_joints > 0:
            self._warmstart_joint_constraints(model, data, problem, x_0, y_0, z_0)
        if limits is not None and limits.model_max_limits_host > 0:
            self._warmstart_limit_constraints(model, data, limits, problem, x_0, y_0, z_0)
        if contacts is not None and contacts.model_max_contacts_host > 0:
            self._warmstart_contact_constraints(model, data, contacts, problem, x_0, y_0, z_0)

    ###
    # Internals - Per-Step Operations
    ###

    def _update_desaxce_correction(self, problem: DualProblem, z: wp.array):
        """
        Launches a kernel to compute the De Saxce correction velocity using the previous dual variables.\n
        The kernel is parallelized over the number of worlds and the maximum number of contacts.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
            z (wp.array): The dual variable array from the previous iteration.\n
                This can either be the acceleration variable `z_hat` or the standard dual variable `z_p`.
        """
        wp.launch(
            kernel=_compute_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.status,
                z,
                # Outputs:
                self._data.state.s,
            ],
        )

    def _update_velocity_bias(self, problem: DualProblem, y: wp.array, z: wp.array):
        """
        Launches a kernel to compute the total bias velocity vector using the previous state variables.\n
        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
            y (wp.array): The primal variable array from the previous iteration.\n
                This can either be the acceleration variable `y_hat` or the standard primal variable `y_p`.
            z (wp.array): The dual variable array from the previous iteration.\n
                This can either be the acceleration variable `z_hat` or the standard dual variable `z_p`.
        """
        wp.launch(
            kernel=_compute_velocity_bias,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.v_f,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.s,
                self._data.state.x_p,
                y,
                z,
                # Outputs:
                self._data.state.v,
            ],
        )

    def _update_unconstrained_solution(self, problem: DualProblem):
        """
        Launches a kernel to solve the unconstrained sub-problem for the primal variables.\n
        For `DualProblem` solves, this corresponds to solving a linear system with the Delassus matrix.\n
        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # TODO: We should do this in-place
        # wp.copy(self._data.state.x, self._data.state.v)
        # problem._delassus.solve_inplace(x=self._data.state.x)
        problem._delassus.solve(v=self._data.state.v, x=self._data.state.x)

    def _update_projection_argument(self, problem: DualProblem, z: wp.array):
        """
        Launches a kernel to compute the argument for the projection operator onto the
        feasible set using the accelerated state variables and the unconstrained solution.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
            z (wp.array): The dual variable array from the previous iteration.\n
                This can either be the acceleration variable `z_hat` or the standard dual variable `z
        """
        # Apply over-relaxation and compute the argument to the projection operator
        wp.launch(
            kernel=_compute_projection_argument,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.penalty,
                self._data.status,
                z,
                self._data.state.x,
                # Outputs:
                self._data.state.y,
            ],
        )

    def _update_projection_to_feasible_set(self, problem: DualProblem):
        """
        Launches a kernel to project the current primal variables
        onto the feasible set defined by the constraint cone K.

        The kernel is parallelized over the number of worlds and the maximum
        number of unilateral constraints, i.e. 1D limits and 3D contacts.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Project to the feasible set defined by the cone K := R^{njd} x R_+^{nld} x K_{mu}^{nc}
        wp.launch(
            kernel=_project_to_feasible_cone,
            dim=(self._size.num_worlds, self._size.max_of_max_unilaterals),
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.status,
                # Outputs:
                self._data.state.y,
            ],
        )

    def _update_complementarity_residuals(self, problem: DualProblem):
        """
        Launches a kernel to compute the complementarity residuals from the current state variables.
        The kernel is parallelized over the number of worlds and the maximum number of unilateral constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute complementarity residual from the current state
        wp.launch(
            kernel=_compute_complementarity_residuals,
            dim=(self._size.num_worlds, self._size.max_of_max_unilaterals),
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.vio,
                problem.data.uio,
                problem.data.lcgo,
                problem.data.ccgo,
                self._data.status,
                self._data.state.x,
                self._data.state.z,
                # Outputs:
                self._data.residuals.r_compl,
            ],
        )

    def _update_dual_variables_and_residuals(self, problem: DualProblem):
        """
        Launches a kernel to update the dual variables and compute the
        PADMM residuals from the current and previous state variables.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Update the dual variables and compute primal-dual residuals from the current state
        # NOTE: These are combined into a single kernel to reduce kernel launch overhead
        wp.launch(
            kernel=self._update_dual_variables_and_compute_primal_dual_residuals_kernel,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.x,
                self._data.state.y,
                self._data.state.x_p,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.z,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_dx,
                self._data.residuals.r_dy,
                self._data.residuals.r_dz,
            ],
        )

        # Compute complementarity residual from the current state
        self._update_complementarity_residuals(problem)

    def _update_dual_variables_and_residuals_accel(self, problem: DualProblem):
        """
        Launches a kernel to update the dual variables and compute the
        PADMM residuals from the current and accelerated state variables.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Update the dual variables and compute primal-dual residuals from the current state
        # NOTE: These are combined into a single kernel to reduce kernel launch overhead
        wp.launch(
            kernel=self._update_dual_variables_and_compute_primal_dual_residuals_kernel,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.x,
                self._data.state.y,
                self._data.state.x_p,
                self._data.state.y_hat,
                self._data.state.z_hat,
                # Outputs:
                self._data.state.z,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_dx,
                self._data.residuals.r_dy,
                self._data.residuals.r_dz,
            ],
        )

        # Compute complementarity residual from the current state
        self._update_complementarity_residuals(problem)

    def _update_convergence_check(self, problem: DualProblem):
        """
        Launches a kernel to compute the infinity-norm of the PADMM residuals
        using the current and previous state variables and check for convergence.

        The kernel is parallelized over the number of worlds.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute infinity-norm of all residuals and check for convergence
        wp.launch(
            kernel=_compute_infnorm_residuals_serially,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.uio,
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_compl,
                # Outputs:
                self._data.state.done,
                self._data.status,
            ],
        )

    def _update_convergence_check_accel(self, problem: DualProblem):
        """
        Launches a kernel to compute the infinity-norm of the PADMM residuals
        using the current and accelerated state variables and check for convergence.

        The kernel is parallelized over the number of worlds.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute infinity-norm of all residuals and check for convergence
        wp.launch(
            kernel=_compute_infnorm_residuals_serially_accel,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.uio,
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.penalty,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_compl,
                self._data.residuals.r_dx,
                self._data.residuals.r_dy,
                self._data.residuals.r_dz,
                self._data.state.a_p,
                # Outputs:
                self._data.state.done,
                self._data.state.a,
                self._data.status,
            ],
        )

    def _update_acceleration(self, problem: DualProblem):
        """
        Launches a kernel to update gradient acceleration and the accelerated state variables.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        wp.launch(
            kernel=_update_state_with_acceleration,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.status,
                self._data.state.a,
                self._data.state.y,
                self._data.state.z,
                self._data.state.a_p,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.y_hat,
                self._data.state.z_hat,
            ],
        )

    def _update_solver_info(self, problem: DualProblem):
        """
        Launches a kernel to update the solver info history from the current solver data.

        The kernel is parallelized over the number of worlds.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # First reset the internal buffer arrays to zero
        # to ensure we do not accumulate values across iterations
        self.data.info.v_plus.zero_()
        self.data.info.v_aug.zero_()
        self._data.info.s.zero_()

        # Collect convergence information from the current state
        wp.launch(
            kernel=self._collect_solver_info_kernel,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.dim,
                problem.data.vio,
                problem.data.mio,
                problem.data.mu,
                problem.data.v_f,
                problem.data.D,
                problem.data.P,
                self._data.state.sigma,
                self._data.state.s,
                self._data.state.x,
                self._data.state.x_p,
                self._data.state.y,
                self._data.state.y_p,
                self._data.state.z,
                self._data.state.z_p,
                self._data.state.a,
                self._data.penalty,
                self._data.status,
                # Outputs:
                self._data.info.lambdas,
                self._data.info.v_plus,
                self._data.info.v_aug,
                self._data.info.s,
                self._data.info.offsets,
                self._data.info.num_restarts,
                self._data.info.num_rho_updates,
                self._data.info.a,
                self._data.info.norm_s,
                self._data.info.norm_x,
                self._data.info.norm_y,
                self._data.info.norm_z,
                self._data.info.f_ccp,
                self._data.info.f_ncp,
                self._data.info.r_dx,
                self._data.info.r_dy,
                self._data.info.r_dz,
                self._data.info.r_primal,
                self._data.info.r_dual,
                self._data.info.r_compl,
                self._data.info.r_pd,
                self._data.info.r_dp,
                self._data.info.r_comb,
                self._data.info.r_comb_ratio,
                self._data.info.r_ncp_primal,
                self._data.info.r_ncp_dual,
                self._data.info.r_ncp_compl,
                self._data.info.r_ncp_natmap,
            ],
        )

    def _update_previous_state(self):
        """
        Updates the cached previous state variables with the current.
        This function uses on-device memory copy operations.
        """
        wp.copy(self._data.state.x_p, self._data.state.x)
        wp.copy(self._data.state.y_p, self._data.state.y)
        wp.copy(self._data.state.z_p, self._data.state.z)

    def _update_previous_state_accel(self):
        """
        Updates the cached previous acceleration and state variable with the current.
        This function uses on-device memory copy operations.
        """
        wp.copy(self._data.state.a_p, self._data.state.a)

        # Cache previous state variables
        self._update_previous_state()

    ###
    # Internals - Post-Solve Operations
    ###

    def _update_solution(self, problem: DualProblem):
        """
        Launches a set of kernels to extract and post-process
        the final solution from the internal PADMM state data.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Apply the dual preconditioner to recover the final PADMM state
        wp.launch(
            kernel=_apply_dual_preconditioner_to_state,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                # Outputs:
                self._data.state.x,
                self._data.state.y,
                self._data.state.z,
            ],
        )

        # Update the De Saxce correction from terminal PADMM dual variables
        wp.launch(
            kernel=_compute_final_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.state.z,
                # Outputs:
                self._data.state.s,
            ],
        )

        # Update solution vectors from the terminal PADMM state
        wp.launch(
            kernel=_compute_solution_vectors,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.state.s,
                self._data.state.y,
                self._data.state.z,
                # Outputs:
                self._data.solution.v_plus,
                self._data.solution.lambdas,
            ],
        )
