# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines the :class:`SolverKaminoImpl` class, providing a physics backend for
simulating constrained multi-body systems for arbitrary mechanical assemblies.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

# Newton imports
from ....core.types import override
from ....sim import Contacts, State
from ...solver import SolverBase

# Kamino imports
from ..solver_kamino import SolverKamino
from .core.bodies import update_body_inertias, update_body_wrenches
from .core.control import ControlKamino
from .core.data import DataKamino
from .core.joints import JointCorrectionMode
from .core.model import ModelKamino
from .core.state import StateKamino
from .core.time import advance_time
from .core.types import float32, int32, transformf, vec6f
from .dynamics.dual import DualProblem
from .dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from .geometry.contacts import ContactsKamino
from .geometry.detector import CollisionDetector
from .integrators import IntegratorEuler, IntegratorMoreauJean
from .kinematics.constraints import (
    make_unilateral_constraints_info,
    unpack_constraint_solutions,
    update_constraints_info,
)
from .kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from .kinematics.joints import (
    compute_joints_data,
    extract_actuators_state_from_joints,
    extract_joints_state_from_actuators,
)
from .kinematics.limits import LimitsKamino
from .kinematics.resets import (
    reset_body_net_wrenches,
    reset_joint_constraint_reactions,
    reset_state_from_base_state,
    reset_state_from_bodies_state,
    reset_state_to_model_default,
    reset_time,
    set_joint_state_masked,
)
from .linalg import ConjugateResidualSolver, IterativeSolver, LinearSolverNameToType
from .solvers.fk import ForwardKinematicsSolver
from .solvers.metrics import SolutionMetrics
from .solvers.padmm import PADMMSolver, PADMMWarmStartMode
from .solvers.warmstart import WarmstarterContacts, WarmstarterLimits
from .utils import logger as msg

###
# Module interface
###

__all__ = [
    "SolverKaminoImpl",
]


###
# Interfaces
###


class SolverKaminoImpl(SolverBase):
    """
    The :class:`SolverKaminoImpl` class implements the core Kamino physics solver.

    This class currently holds the actual implementation of the solver, and is wrapped
    by the upper-level :class:`SolverKamino` class which serves as the main user-facing
    API for now. In the future, a complete refactoring of Kamino will integrate the solver
    and underlying components with Newton end-to-end and completely. At that point, the
    :class:`SolverKaminoImpl` class will be removed and its implementation merged into
    the :class:`SolverKamino` class.
    """

    Config = SolverKamino.Config
    """
    Defines a type alias of the PADMM solver configurations container, including convergence
    criteria, maximum iterations, and options for the linear solver and preconditioning.

    See :class:`PADMMSolverConfig` for the full list of configuration options and their descriptions.
    """

    ResetCallbackType = Callable[["SolverKaminoImpl", StateKamino], None]
    """Defines the type signature for reset callback functions."""

    StepCallbackType = Callable[["SolverKaminoImpl", StateKamino, StateKamino, ControlKamino, ContactsKamino], None]
    """Defines the type signature for step callback functions."""

    def __init__(
        self,
        model: ModelKamino,
        contacts: ContactsKamino | None = None,
        config: SolverKaminoImpl.Config | None = None,
    ):
        """
        Initializes the Kamino physics solver for the given set of multi-body systems
        defined in `model`, and the total contact allocations defined in `contacts`.

        Explicit solver config may be provided through the `config` argument. If no
        config is provided, a default config will be used.

        Args:
            model (ModelKamino): The multi-body systems model to simulate.
            contacts (ContactsKamino): The contact data container for the simulation.
            config (SolverKaminoImpl.Config | None): Optional solver config.
        """
        # Ensure the input containers are valid
        if not isinstance(model, ModelKamino):
            raise TypeError(f"Invalid model container: Expected a `ModelKamino` instance, but got {type(model)}.")
        if contacts is not None and not isinstance(contacts, ContactsKamino):
            raise TypeError(
                f"Invalid contacts container: Expected a `ContactsKamino` instance, but got {type(contacts)}."
            )
        if config is not None and not isinstance(config, SolverKaminoImpl.Config):
            raise TypeError(
                f"Invalid solver config: Expected a `SolverKaminoImpl.Config` instance, but got {type(config)}."
            )

        # First initialize the base solver
        # NOTE: Although we pass the model here, we will re-assign it below
        # since currently Kamino defines its own :class`ModelKamino` class.
        super().__init__(model=model)
        self._model = model

        # If no explicit config is provided, attempt to create a config
        # from the model attributes (e.g. if imported from USD assets).
        # NOTE: `Config.from_model` will default-initialize if no relevant custom attributes were
        # found on the model, so `self._config` will always be fully initialized after this step.
        if config is None:
            config = self.Config.from_model(model._model)

        # Validate the solver configurations and raise errors early if invalid
        config.validate()

        # Cache the solver config and parse relevant options for internal use
        self._config: SolverKaminoImpl.Config = config
        self._warmstart_mode: PADMMWarmStartMode = PADMMWarmStartMode.from_string(config.padmm.warmstart_mode)
        self._rotation_correction: JointCorrectionMode = JointCorrectionMode.from_string(config.rotation_correction)

        # ---------------------------------------------------------------------------
        # TODO: Migrate this entire section into the constructor of `DualProblem`

        # Convert the linear solver type from the config literal to the concrete class, raising an error if invalid
        linear_solver_type = LinearSolverNameToType.get(self._config.dynamics.linear_solver_type, None)
        if linear_solver_type is None:
            raise ValueError(
                "Invalid linear solver type: Expected one of "
                f"{list(LinearSolverNameToType.keys())}, got '{linear_solver_type}'."
            )

        # Override the linear solver type to an iterative solver if
        # sparsity is enabled but the provided solver is not iterative
        if self._config.sparse_dynamics and not issubclass(linear_solver_type, IterativeSolver):
            msg.warning(
                f"Sparse dynamics requires an iterative solver, but got '{linear_solver_type.__name__}'."
                " Defaulting to 'ConjugateResidualSolver' as the PADMM linear solver."
            )
            linear_solver_type = ConjugateResidualSolver

        # If graph conditionals are disabled in the PADMM solver, ensure that they
        # are also disabled in the linear solver if it is an iterative solver.
        linear_solver_kwargs = dict(self._config.dynamics.linear_solver_kwargs)
        if not self._config.padmm.use_graph_conditionals and issubclass(linear_solver_type, IterativeSolver):
            linear_solver_kwargs.setdefault("use_graph_conditionals", False)

        # Bundle both constraint stabilization and forward-
        # dynamics problem configurations into a single object
        problem_fd_config = DualProblem.Config(
            constraints=self._config.constraints,
            dynamics=self._config.dynamics,
            # TODO: linear_solver_type=linear_solver_type,
            # TODO: linear_solver_kwargs=linear_solver_kwargs,
            # TODO: sparse=bool(self._config.sparse_dynamics),
        )

        # ---------------------------------------------------------------------------

        # Allocate internal time-varying solver data
        self._data = self._model.data()

        # Allocate a joint-limits interface
        self._limits = LimitsKamino(model=self._model)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(model=self._model, data=self._data, limits=self._limits, contacts=contacts)

        # Allocate Jacobians data on the device
        if self._config.sparse_jacobian:
            self._jacobians = SparseSystemJacobians(
                model=self._model,
                limits=self._limits,
                contacts=contacts,
            )
        else:
            self._jacobians = DenseSystemJacobians(
                model=self._model,
                limits=self._limits,
                contacts=contacts,
            )

        # Allocate the dual problem data on the device
        self._problem_fd = DualProblem(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            jacobians=self._jacobians,
            config=problem_fd_config,
            solver=linear_solver_type,
            solver_kwargs=linear_solver_kwargs,
            sparse=self._config.sparse_dynamics,
        )

        # Allocate the forward dynamics solver on the device
        self._solver_fd = PADMMSolver(
            model=self._model,
            config=self._config.padmm,
            warmstart=self._warmstart_mode,
            use_acceleration=self._config.padmm.use_acceleration,
            use_graph_conditionals=self._config.padmm.use_graph_conditionals,
            collect_info=self._config.collect_solver_info,
        )

        # Allocate the forward kinematics solver on the device
        self._solver_fk = None
        if self._config.use_fk_solver:
            self._solver_fk = ForwardKinematicsSolver(model=self._model, config=self._config.fk)

        # Create the time-integrator instance based on the config
        if self._config.integrator == "euler":
            self._integrator = IntegratorEuler(model=self._model)
        elif self._config.integrator == "moreau":
            self._integrator = IntegratorMoreauJean(model=self._model)
        else:
            raise ValueError(
                f"Unsupported integrator type: Expected 'euler' or 'moreau', but got {self._config.integrator}."
            )

        # Allocate additional internal data for reset operations
        with wp.ScopedDevice(self._model.device):
            self._all_worlds_mask = wp.ones(shape=(self._model.size.num_worlds,), dtype=int32)
            self._base_q = wp.zeros(shape=(self._model.size.num_worlds,), dtype=transformf)
            self._base_u = wp.zeros(shape=(self._model.size.num_worlds,), dtype=vec6f)
            self._bodies_u_zeros = wp.zeros(shape=(self._model.size.sum_of_num_bodies,), dtype=vec6f)
            self._actuators_q = wp.zeros(shape=(self._model.size.sum_of_num_actuated_joint_coords,), dtype=float32)
            self._actuators_u = wp.zeros(shape=(self._model.size.sum_of_num_actuated_joint_dofs,), dtype=float32)

        # Allocate the contacts warmstarter if enabled
        self._ws_limits: WarmstarterLimits | None = None
        self._ws_contacts: WarmstarterContacts | None = None
        if self._warmstart_mode == PADMMWarmStartMode.CONTAINERS:
            self._ws_limits = WarmstarterLimits(limits=self._limits)
            self._ws_contacts = WarmstarterContacts(
                contacts=contacts,
                method=WarmstarterContacts.Method.from_string(self._config.padmm.contact_warmstart_method),
            )

        # Allocate the solution metrics evaluator if enabled
        self._metrics: SolutionMetrics | None = None
        if self._config.compute_solution_metrics:
            self._metrics = SolutionMetrics(model=self._model)

        # Initialize callbacks
        self._pre_reset_cb: SolverKaminoImpl.ResetCallbackType | None = None
        self._post_reset_cb: SolverKaminoImpl.ResetCallbackType | None = None
        self._pre_step_cb: SolverKaminoImpl.StepCallbackType | None = None
        self._mid_step_cb: SolverKaminoImpl.StepCallbackType | None = None
        self._post_step_cb: SolverKaminoImpl.StepCallbackType | None = None

        # Initialize all internal solver data
        with wp.ScopedDevice(self._model.device):
            self._reset()

    ###
    # Properties
    ###

    @property
    def config(self) -> SolverKaminoImpl.Config:
        """
        Returns the host-side cache of high-level solver config.
        """
        return self._config

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device where the solver data is allocated.
        """
        return self._model.device

    @property
    def data(self) -> DataKamino:
        """
        Returns the internal solver data container.
        """
        return self._data

    @property
    def problem_fd(self) -> DualProblem:
        """
        Returns the dual forward dynamics problem.
        """
        return self._problem_fd

    @property
    def solver_fd(self) -> PADMMSolver:
        """
        Returns the forward dynamics solver.
        """
        return self._solver_fd

    @property
    def solver_fk(self) -> ForwardKinematicsSolver | None:
        """
        Returns the forward kinematics solver backend, if it was initialized.
        """
        return self._solver_fk

    @property
    def metrics(self) -> SolutionMetrics | None:
        """
        Returns the solution metrics evaluator, if enabled.
        """
        return self._metrics

    ###
    # Configurations
    ###

    def set_pre_reset_callback(self, callback: ResetCallbackType):
        """
        Set a reset callback to be called at the beginning of each call to `reset_*()` methods.
        """
        self._pre_reset_cb = callback

    def set_post_reset_callback(self, callback: ResetCallbackType):
        """
        Set a reset callback to be called at the end of each call to to `reset_*()` methods.
        """
        self._post_reset_cb = callback

    def set_pre_step_callback(self, callback: StepCallbackType):
        """
        Sets a callback to be called before forward dynamics solve.
        """
        self._pre_step_cb = callback

    def set_mid_step_callback(self, callback: StepCallbackType):
        """
        Sets a callback to be called between forward dynamics solver and state integration.
        """
        self._mid_step_cb = callback

    def set_post_step_callback(self, callback: StepCallbackType):
        """
        Sets a callback to be called after state integration.
        """
        self._post_step_cb = callback

    ###
    # Solver API
    ###

    def reset(
        self,
        state_out: StateKamino,
        world_mask: wp.array | None = None,
        actuator_q: wp.array | None = None,
        actuator_u: wp.array | None = None,
        joint_q: wp.array | None = None,
        joint_u: wp.array | None = None,
        base_q: wp.array | None = None,
        base_u: wp.array | None = None,
        bodies_q: wp.array | None = None,
        bodies_u: wp.array | None = None,
    ):
        """
        Resets the simulation state given a combination of desired base body
        and joint states, as well as an optional per-world mask array indicating
        which worlds should be reset. The reset state is written to `state_out`.

        For resets given absolute quantities like base body poses, the
        `state_out` must initially contain the current state of the simulation.

        Args:
            state_out (StateKamino):
                The output state container to which the reset state data is written.
            world_mask (wp.array, optional):
                Optional array of per-world masks indicating which worlds should be reset.\n
                Shape of `(num_worlds,)` and type :class:`wp.int8 | wp.bool`
            actuator_q (wp.array, optional):
                Optional array of target actuated joint coordinates.\n
                Shape of `(num_actuated_joint_coords,)` and type :class:`wp.float32`
            actuator_u (wp.array, optional):
                Optional array of target actuated joint DoF velocities.\n
                Shape of `(num_actuated_joint_dofs,)` and type :class:`wp.float32`
            joint_q (wp.array, optional):
                Optional array of target joint coordinates.\n
                Shape of `(num_joint_coords,)` and type :class:`wp.float32`
            joint_u (wp.array, optional):
                Optional array of target joint DoF velocities.\n
                Shape of `(num_joint_dofs,)` and type :class:`wp.float32`
            base_q (wp.array, optional):
                Optional array of target base body poses.\n
                Shape of `(num_worlds,)` and type :class:`wp.transformf`
            base_u (wp.array, optional):
                Optional array of target base body twists.\n
                Shape of `(num_worlds,)` and type :class:`wp.spatial_vectorf`
            bodies_q (wp.array, optional):
                Optional array of target body poses.\n
                Shape of `(num_bodies,)` and type :class:`wp.transformf`
            bodies_u (wp.array, optional):
                Optional array of target body twists.\n
                Shape of `(num_bodies,)` and type :class:`wp.spatial_vectorf`
        """

        # Ensure the input reset targets are valid
        def _check_length(data: wp.array, name: str, expected: int):
            if data is not None and data.shape[0] != expected:
                raise ValueError(f"Invalid {name} shape: Expected ({expected},), but got {data.shape}.")

        _check_length(joint_q, "joint_q", self._model.size.sum_of_num_joint_coords)
        _check_length(joint_u, "joint_u", self._model.size.sum_of_num_joint_dofs)
        _check_length(actuator_q, "actuator_q", self._model.size.sum_of_num_actuated_joint_coords)
        _check_length(actuator_u, "actuator_u", self._model.size.sum_of_num_actuated_joint_dofs)
        _check_length(base_q, "base_q", self._model.size.num_worlds)
        _check_length(base_u, "base_u", self._model.size.num_worlds)
        _check_length(bodies_q, "bodies_q", self._model.size.sum_of_num_bodies)
        _check_length(bodies_u, "bodies_u", self._model.size.sum_of_num_bodies)
        _check_length(world_mask, "world_mask", self._model.size.num_worlds)

        # Ensure that only joint or actuator targets are provided
        if (joint_q is not None or joint_u is not None) and (actuator_q is not None or actuator_u is not None):
            raise ValueError("Combined joint and actuator targets are not supported. Only one type may be provided.")

        # Ensure that joint/actuator velocity-only resets are prevented
        if (joint_q is None and joint_u is not None) or (actuator_q is None and actuator_u is not None):
            raise ValueError("Velocity-only joint or actuator resets are not supported.")

        # Run the pre-reset callback if it has been set
        self._run_pre_reset_callback(state_out=state_out)

        # Determine the effective world mask to use for the reset operation
        _world_mask = world_mask if world_mask is not None else self._all_worlds_mask

        # Detect mode
        base_reset = base_q is not None or base_u is not None
        joint_reset = joint_q is not None or actuator_q is not None
        bodies_reset = bodies_q is not None or bodies_u is not None

        # If no reset targets are provided, reset all bodies to the model default state
        if not base_reset and not joint_reset and not bodies_reset:
            self._reset_to_default_state(
                state_out=state_out,
                world_mask=_world_mask,
            )

        # If only base targets are provided, uniformly reset all bodies to the given base states
        elif base_reset and not joint_reset and not bodies_reset:
            self._reset_to_base_state(
                state_out=state_out,
                world_mask=_world_mask,
                base_q=base_q,
                base_u=base_u,
            )

        # If a joint target is provided, use the FK solver to reset the bodies accordingly
        elif joint_reset and not bodies_reset:
            self._reset_with_fk_solve(
                state_out=state_out,
                world_mask=_world_mask,
                actuator_q=actuator_q,
                actuator_u=actuator_u,
                joint_q=joint_q,
                joint_u=joint_u,
                base_q=base_q,
                base_u=base_u,
            )

        # If body targets are provided, reset bodies directly
        elif not base_reset and not joint_reset and bodies_reset:
            self._reset_to_bodies_state(
                state_out=state_out,
                world_mask=_world_mask,
                bodies_q=bodies_q,
                bodies_u=bodies_u,
            )

        # If no valid combination of reset targets is provided, raise an error
        else:
            raise ValueError(
                "Unsupported reset combination with: "
                f" actuator_q: {actuator_q is not None}, actuator_u: {actuator_u is not None},"
                f" joint_q: {joint_q is not None}, joint_u: {joint_u is not None},"
                f" base_q: {base_q is not None}, base_u: {base_u is not None}."
                f" bodies_q: {bodies_q is not None}, bodies_u: {bodies_u is not None}."
            )

        # Post-process the reset operation
        self._reset_post_process(world_mask=_world_mask)

        # Run the post-reset callback if it has been set
        self._run_post_reset_callback(state_out=state_out)

    @override
    def step(
        self,
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        contacts: ContactsKamino | None = None,
        detector: CollisionDetector | None = None,
        dt: float | None = None,
    ):
        """
        Progresses the simulation by a single time-step `dt` given the current
        state `state_in`, control inputs `control`, and set of active contacts
        `contacts`. The updated state is written to `state_out`.

        Args:
            state_in (StateKamino):
                The input current state of the simulation.
            state_out (StateKamino):
                The output next state after time integration.
            control (ControlKamino):
                The input controls applied to the system.
            contacts (ContactsKamino, optional):
                The set of active contacts.
            detector (CollisionDetector, optional):
                An optional collision detector to use for generating contacts at the current state.\n
                If `None`, the `contacts` data will be used as the current set of active contacts.
            dt (float, optional):
                A uniform time-step to apply uniformly to all worlds of the simulation.
        """
        # If specified, configure the internal per-world solver time-step uniformly from the input argument
        if dt is not None:
            self._model.time.set_uniform_timestep(dt)

        # Copy the new input state and control to the internal solver data
        self._read_step_inputs(state_in=state_in, control_in=control)

        # Execute state integration:
        #  - Optionally calls limit and contact detection to generate unilateral constraints
        #  - Solves the forward dynamics sub-problem to compute constraint reactions
        #  - Integrates the state forward in time
        self._integrator.integrate(
            forward=self._solve_forward_dynamics,
            model=self._model,
            data=self._data,
            state_in=state_in,
            state_out=state_out,
            control=control,
            limits=self._limits,
            contacts=contacts,
            detector=detector,
        )

        # Update the internal joint states from the
        # updated body states after time-integration
        self._update_joints_data()

        # Compute solver solution metrics if enabled
        self._compute_metrics(state_in=state_in, contacts=contacts)

        # Update time-keeping (i.e. physical time and discrete steps)
        self._advance_time()

        # Run the post-step callback if it has been set
        self._run_poststep_callback(state_in, state_out, control, contacts)

        # Copy the updated internal solver state to the output state
        self._write_step_output(state_out=state_out)

    @override
    def notify_model_changed(self, flags: int) -> None:
        pass  # TODO: Migrate implementation when we fully integrate with Newton

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        pass  # TODO: Migrate implementation when we fully integrate with Newton

    @override
    @classmethod
    def register_custom_attributes(cls, flags: int):
        pass  # TODO: Migrate implementation when we fully integrate with Newton

    ###
    # Internals - Callback Operations
    ###

    def _run_pre_reset_callback(self, state_out: StateKamino):
        """
        Runs the pre-reset callback if it has been set.
        """
        if self._pre_reset_cb is not None:
            self._pre_reset_cb(self, state_out)

    def _run_post_reset_callback(self, state_out: StateKamino):
        """
        Runs the post-reset callback if it has been set.
        """
        if self._post_reset_cb is not None:
            self._post_reset_cb(self, state_out)

    def _run_prestep_callback(
        self, state_in: StateKamino, state_out: StateKamino, control: ControlKamino, contacts: ContactsKamino
    ):
        """
        Runs the pre-step callback if it has been set.
        """
        if self._pre_step_cb is not None:
            self._pre_step_cb(self, state_in, state_out, control, contacts)

    def _run_midstep_callback(
        self, state_in: StateKamino, state_out: StateKamino, control: ControlKamino, contacts: ContactsKamino
    ):
        """
        Runs the mid-step callback if it has been set.
        """
        if self._mid_step_cb is not None:
            self._mid_step_cb(self, state_in, state_out, control, contacts)

    def _run_poststep_callback(
        self, state_in: StateKamino, state_out: StateKamino, control: ControlKamino, contacts: ContactsKamino
    ):
        """
        Executes the post-step callback if it has been set.
        """
        if self._post_step_cb is not None:
            self._post_step_cb(self, state_in, state_out, control, contacts)

    ###
    # Internals - Input/Output Operations
    ###

    def _read_step_inputs(self, state_in: StateKamino, control_in: ControlKamino):
        """
        Updates the internal solver data from the input state and control.

        Control inputs (tau_j, q_j_ref, dq_j_ref, tau_j_ref) are aliased
        directly to avoid redundant device-to-device copies since they are
        only read during a step. State arrays must still be copied because
        the solver modifies them in-place.
        """
        # TODO: Remove corresponding data copies
        # by directly using the input containers
        wp.copy(self._data.bodies.q_i, state_in.q_i)
        wp.copy(self._data.bodies.u_i, state_in.u_i)
        wp.copy(self._data.bodies.w_i, state_in.w_i)
        wp.copy(self._data.bodies.w_e_i, state_in.w_i_e)
        wp.copy(self._data.joints.q_j, state_in.q_j)
        wp.copy(self._data.joints.q_j_p, state_in.q_j_p)
        wp.copy(self._data.joints.dq_j, state_in.dq_j)
        wp.copy(self._data.joints.lambda_j, state_in.lambda_j)
        # Alias read-only control inputs
        self._data.joints.tau_j = control_in.tau_j
        self._data.joints.q_j_ref = control_in.q_j_ref
        self._data.joints.dq_j_ref = control_in.dq_j_ref
        self._data.joints.tau_j_ref = control_in.tau_j_ref

    def _write_step_output(self, state_out: StateKamino):
        """
        Updates the output state from the internal solver data.
        """
        # TODO: Remove corresponding data copies
        # by directly using the input containers
        wp.copy(state_out.q_i, self._data.bodies.q_i)
        wp.copy(state_out.u_i, self._data.bodies.u_i)
        wp.copy(state_out.w_i, self._data.bodies.w_i)
        wp.copy(state_out.w_i_e, self._data.bodies.w_e_i)
        wp.copy(state_out.q_j, self._data.joints.q_j)
        wp.copy(state_out.q_j_p, self._data.joints.q_j_p)
        wp.copy(state_out.dq_j, self._data.joints.dq_j)
        wp.copy(state_out.lambda_j, self._data.joints.lambda_j)

    ###
    # Internals - Reset Operations
    ###

    def _reset(self):
        """
        Performs a hard-reset of all solver internal data.
        """
        # Reset internal time-keeping data
        self._data.time.reset()

        # Reset all bodies to their model default states
        self._data.bodies.clear_all_wrenches()
        wp.copy(self._data.bodies.q_i, self._model.bodies.q_i_0)
        wp.copy(self._data.bodies.u_i, self._model.bodies.u_i_0)
        update_body_inertias(model=self._model.bodies, data=self._data.bodies)

        # Reset all joints to their model default states
        self._data.joints.reset_state(q_j_0=self._model.joints.q_j_0)
        self._data.joints.clear_all()

        # Reset the joint-limits interface
        self._limits.reset()

        # Initialize the constraint state info
        self._data.info.num_limits.zero_()
        self._data.info.num_contacts.zero_()
        update_constraints_info(model=self._model, data=self._data)

        # Initialize the system Jacobians so that they may be available after reset
        # NOTE: This is not strictly necessary, but serves advanced users who may
        # want to query Jacobians in controllers immediately after a reset operation.
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=None,
            contacts=None,
            reset_to_zero=True,
        )

        # Reset the forward dynamics solver
        self._solver_fd.reset()

    def _reset_to_default_state(self, state_out: StateKamino, world_mask: wp.array):
        """
        Resets the simulation to the default state defined in the model.
        """
        reset_state_to_model_default(
            model=self._model,
            state_out=state_out,
            world_mask=world_mask,
        )

    def _reset_to_base_state(
        self,
        state_out: StateKamino,
        world_mask: wp.array,
        base_q: wp.array,
        base_u: wp.array | None = None,
    ):
        """
        Resets the simulation to the given base body states by
        uniformly applying the necessary transform across all bodies.
        """
        # Ensure that the base pose reset targets are valid
        if base_q is None:
            raise ValueError("Base pose targets must be provided for base state resets.")
        if base_q.shape[0] != self._model.size.num_worlds:
            raise ValueError(
                f"Invalid base_q shape: Expected ({self._model.size.num_worlds},), but got {base_q.shape}."
            )

        # Determine the effective base twists to use
        _base_u = base_u if base_u is not None else self._base_u

        # Uniformly reset all bodies according to the transform between the given
        # base state and the existing body states contained in `state_out`
        reset_state_from_base_state(
            model=self._model,
            state_out=state_out,
            world_mask=world_mask,
            base_q=base_q,
            base_u=_base_u,
        )

    def _reset_to_bodies_state(
        self,
        state_out: StateKamino,
        world_mask: wp.array,
        bodies_q: wp.array | None = None,
        bodies_u: wp.array | None = None,
    ):
        """
        Resets the simulation to the given rigid body states.
        There is no check that the provided states satisfy any kinematic constraints.
        """

        # use initial model poses if not provided
        _bodies_q = bodies_q if bodies_q is not None else self._model.bodies.q_i_0
        # use zero body velocities if not provided
        _bodies_u = bodies_u if bodies_u is not None else self._bodies_u_zeros

        reset_state_from_bodies_state(
            model=self._model,
            state_out=state_out,
            world_mask=world_mask,
            bodies_q=_bodies_q,
            bodies_u=_bodies_u,
        )

    def _reset_with_fk_solve(
        self,
        state_out: StateKamino,
        world_mask: wp.array,
        joint_q: wp.array | None = None,
        joint_u: wp.array | None = None,
        actuator_q: wp.array | None = None,
        actuator_u: wp.array | None = None,
        base_q: wp.array | None = None,
        base_u: wp.array | None = None,
    ):
        """
        Resets the simulation to the given joint states by solving
        the forward kinematics to compute the corresponding body states.
        """
        # Check that the FK solver was initialized
        if self._solver_fk is None:
            raise RuntimeError("The FK solver must be enabled to use resets from joint angles.")

        # Detect if joint or actuator targets are provided
        with_joint_targets = joint_q is not None and (actuator_q is None and actuator_u is None)

        # Unpack the actuated joint states from the input joint states
        if with_joint_targets:
            extract_actuators_state_from_joints(
                model=self._model,
                world_mask=world_mask,
                joint_q=joint_q,
                joint_u=joint_u if joint_u is not None else state_out.dq_j,
                actuator_q=self._actuators_q,
                actuator_u=self._actuators_u,
            )

        # Determine the actuator state arrays to use for the FK solve
        _actuator_q = actuator_q if actuator_q is not None else self._actuators_q
        _actuator_u = actuator_u if actuator_u is not None else self._actuators_u

        # TODO: We need a graph-capturable mechanism to detect solver errors
        # Solve the forward kinematics to compute the body states
        self._solver_fk.run_fk_solve(
            world_mask=world_mask,
            bodies_q=state_out.q_i,
            bodies_u=state_out.u_i if joint_u is not None or actuator_u is not None else None,
            actuators_q=_actuator_q,
            actuators_u=_actuator_u,
            base_q=base_q,
            base_u=base_u,
        )

        # Reset net body wrenches and joint constraint reactions to zero
        # NOTE: This is necessary to ensure proper solver behavior after resets
        reset_body_net_wrenches(model=self._model, body_w=state_out.w_i, world_mask=world_mask)
        reset_joint_constraint_reactions(model=self._model, lambda_j=state_out.lambda_j, world_mask=world_mask)

        # If joint targets were provided, write them to the output state. The mask-aware
        # write ensures worlds outside `world_mask` keep their previous values (notably
        # `q_j_p`, the TWOPI angle-correction reference).
        if with_joint_targets:
            set_joint_state_masked(
                model=self._model,
                world_mask=world_mask,
                src_q=joint_q,
                src_u=joint_u,
                dst_q=state_out.q_j,
                dst_q_p=state_out.q_j_p,
                dst_dq=state_out.dq_j,
            )
        # Otherwise, extract the joint states from the actuators and synchronize `q_j_p`
        else:
            extract_joints_state_from_actuators(
                model=self._model,
                world_mask=world_mask,
                actuator_q=_actuator_q,
                actuator_u=_actuator_u,
                joint_q=state_out.q_j,
                joint_u=state_out.dq_j,
            )
            set_joint_state_masked(
                model=self._model,
                world_mask=world_mask,
                src_q=state_out.q_j,
                src_u=None,
                dst_q=state_out.q_j,
                dst_q_p=state_out.q_j_p,
                dst_dq=None,
            )

    def _reset_post_process(self, world_mask: wp.array | None = None):
        """
        Resets solver internal data and calls reset callbacks.

        This is a common operation that must be called after resetting bodies and joints,
        that ensures that all state and control data are synchronized with the internal
        solver state, and that intermediate quantities are updated accordingly.
        """
        # Reset the solver-internal time-keeping data
        reset_time(
            model=self._model,
            world_mask=world_mask,
            time=self._data.time.time,
            steps=self._data.time.steps,
        )

        # Reset the forward dynamics solver to clear internal state
        # NOTE: This will cause the solver to perform a cold-start
        # on the first call to `step()`
        self._solver_fd.reset(problem=self._problem_fd, world_mask=world_mask)

        # TODO: Enable this when world-masking is implemented
        # Reset the warm-starting caches if enabled
        # if self._warmstart_mode == PADMMWarmStartMode.CONTAINERS:
        #     self._ws_limits.reset()
        #     self._ws_contacts.reset()

    ###
    # Internals - Step Operations
    ###

    def _update_joints_data(self, q_j_p: wp.array | None = None):
        """
        Updates the joint states based on the current body states.
        """
        # Use the provided previous joint states if given,
        # otherwise use the internal cached joint states
        if q_j_p is not None:
            _q_j_p = q_j_p
        else:
            wp.copy(self._data.joints.q_j_p, self._data.joints.q_j)
            _q_j_p = self._data.joints.q_j_p

        # Update the joint states based on the updated body states
        # NOTE: We use the previous state `state_p` for post-processing
        # purposes, e.g. account for roll-over of revolute joints etc
        compute_joints_data(
            model=self._model,
            data=self._data,
            q_j_p=_q_j_p,
            correction=self._rotation_correction,
        )

    def _update_intermediates(self, state_in: StateKamino):
        """
        Updates intermediate quantities required for the forward dynamics solve.
        """
        self._update_joints_data(q_j_p=state_in.q_j_p)
        update_body_inertias(model=self._model.bodies, data=self._data.bodies)

    def _update_limits(self):
        """
        Runs limit detection to generate active joint limits.
        """
        self._limits.detect(q_j=self._data.joints.q_j)

    def _update_constraint_info(self):
        """
        Updates the state info with the set of active constraints resulting from limit and collision detection.
        """
        update_constraints_info(model=self._model, data=self._data)

    def _update_jacobians(self, contacts: ContactsKamino | None = None):
        """
        Updates the forward kinematics by building the system Jacobians (of actuation and
        constraints) based on the current state of the system and set of active constraints.
        """
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            reset_to_zero=True,
        )

    def _update_actuation_wrenches(self):
        """
        Updates the actuation wrenches based on the current control inputs.
        """
        compute_joint_dof_body_wrenches(self._model, self._data, self._jacobians)

    def _update_dynamics(self, contacts: ContactsKamino | None = None):
        """
        Constructs the forward dynamics problem quantities based on the current state of
        the system, the set of active constraints, and the updated system Jacobians.
        """
        self._problem_fd.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            jacobians=self._jacobians,
            reset_to_zero=True,
        )

    def _update_constraints(self, contacts: ContactsKamino | None = None):
        """
        Solves the forward dynamics sub-problem to compute constraint
        reactions and body wrenches effected through constraints.
        """
        # If warm-starting is enabled, initialize unilateral
        # constraints containers from the current solver data
        if self._warmstart_mode > PADMMWarmStartMode.NONE:
            if self._warmstart_mode == PADMMWarmStartMode.CONTAINERS:
                self._ws_limits.warmstart(self._limits)
                self._ws_contacts.warmstart(self._model, self._data, contacts)
            self._solver_fd.warmstart(
                problem=self._problem_fd,
                model=self._model,
                data=self._data,
                limits=self._limits,
                contacts=contacts,
            )
        # Otherwise, perform a cold-start of the dynamics solver
        else:
            self._solver_fd.coldstart()

        # Solve the dual problem to compute the constraint reactions
        self._solver_fd.solve(problem=self._problem_fd)

        # Compute the effective body wrenches applied by the set of
        # active constraints from the respective reaction multipliers
        compute_constraint_body_wrenches(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            jacobians=self._jacobians,
            lambdas_offsets=self._problem_fd.data.vio,
            lambdas_data=self._solver_fd.data.solution.lambdas,
        )

        # Unpack the computed constraint multipliers to the respective joint-limit
        # and contact data for post-processing and optional solver warm-starting
        unpack_constraint_solutions(
            lambdas=self._solver_fd.data.solution.lambdas,
            v_plus=self._solver_fd.data.solution.v_plus,
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
        )

        # If warmstarting is enabled, update the limits and contacts caches
        # with the constraint reactions generated by the dynamics solver
        # NOTE: This needs to happen after unpacking the multipliers
        if self._warmstart_mode == PADMMWarmStartMode.CONTAINERS:
            self._ws_limits.update(self._limits)
            self._ws_contacts.update(contacts)

    def _update_wrenches(self):
        """
        Computes the total (i.e. net) body wrenches by summing up all individual contributions,
        from joint actuation, joint limits, contacts, and purely external effects.
        """
        update_body_wrenches(self._model.bodies, self._data.bodies)

    def _forward(self, contacts: ContactsKamino | None = None):
        """
        Solves the forward dynamics sub-problem to compute constraint reactions
        and total effective body wrenches applied to each body of the system.
        """
        # Update the dynamics
        self._update_dynamics(contacts=contacts)

        # Compute constraint reactions
        self._update_constraints(contacts=contacts)

        # Post-processing
        self._update_wrenches()

    def _solve_forward_dynamics(
        self,
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        detector: CollisionDetector | None = None,
    ):
        """
        Solves the forward dynamics sub-problem to compute constraint reactions
        and total effective body wrenches applied to each body of the system.

        Args:
            state_in (`StateKamino`):
                State of the system at the current time-step.
            state_out (`StateKamino`):
                State of the system at the next time-step.
            control (`ControlKamino`):
                Input controls applied to the system.
            limits (`LimitsKamino`, optional):
                Optional container for joint limits.
                If `None`, joint limit handling is skipped.
            contacts (`ContactsKamino`, optional):
                Optional container of active contacts.
                If `None`, the solver will use the internal collision detector
                if the model admits contacts, or skip contact handling if not.
            detector (`CollisionDetector`, optional):
                Optional collision detector.
                If `None`, collision detection is skipped.
        """
        # Update intermediate quantities of the bodies and joints
        # NOTE: We update the intermediate joint and body data here
        # to ensure that they consistent with the current state.
        # This is to handle cases when the forward dynamics may be
        # evaluated at intermediate points of the discrete time-step
        # (and potentially multiple times). The intermediate data is
        # then used to perform limit and contact detection, as well
        # as to evaluate kinematics and dynamics quantities such as
        # the system Jacobians and generalized mass matrix.
        self._update_intermediates(state_in=state_in)

        # If a collision detector is provided, use it to generate
        # update the set of active contacts at the current state
        if detector is not None:
            detector.collide(data=self._data, state=state_in, contacts=contacts)

        # If a limits container/detector is provided, run joint-limit
        # detection to generate active joint limits at the current state
        if limits is not None:
            limits.detect(q_j=self._data.joints.q_j)

        # Update the constraint state info
        self._update_constraint_info()

        # Update the differential forward kinematics to compute system Jacobians
        self._update_jacobians(contacts=contacts)

        # Compute the body actuation wrenches based on the current control inputs
        self._update_actuation_wrenches()

        # Run the pre-step callback if it has been set
        self._run_prestep_callback(state_in, state_out, control, contacts)

        # Solve the forward dynamics sub-problem to compute constraint reactions and body wrenches
        self._forward(contacts=contacts)

        # Run the mid-step callback if it has been set
        self._run_midstep_callback(state_in, state_out, control, contacts)

    def _compute_metrics(self, state_in: StateKamino, contacts: ContactsKamino | None = None):
        """
        Computes performance metrics measuring the physical fidelity of the dynamics solver solution.
        """
        if self._config.compute_solution_metrics:
            self.metrics.reset()
            self._metrics.evaluate(
                sigma=self._solver_fd.data.state.sigma,
                lambdas=self._solver_fd.data.solution.lambdas,
                v_plus=self._solver_fd.data.solution.v_plus,
                model=self._model,
                data=self._data,
                state_p=state_in,
                problem=self._problem_fd,
                jacobians=self._jacobians,
                limits=self._limits,
                contacts=contacts,
            )

    def _advance_time(self):
        """
        Updates simulation time-keeping (i.e. physical time and discrete steps).
        """
        advance_time(self._model.time, self._data.time)
