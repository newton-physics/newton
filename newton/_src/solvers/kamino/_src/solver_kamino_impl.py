# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines the :class:`SolverKaminoImpl` class, providing a physics backend for
simulating constrained multi-body systems for arbitrary mechanical assemblies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import warp as wp

# Newton imports
from ....core.types import override
from ....sim import Contacts, ModelFlags, State
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
)
from .kinematics.limits import LimitsKamino
from .kinematics.resets import (
    get_base_q_from_joint_q_and_body_q,
    get_base_u_from_joint_u_and_body_u,
    reset_body_velocities,
    reset_body_wrenches,
    reset_joints_state_from_bodies_state,
    reset_time,
    set_body_q,
    set_floating_base,
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
            model: The multi-body systems model to simulate.
            contacts: The contact data container for the simulation.
            config: Optional solver config.
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
            self._all_worlds_mask = wp.ones(shape=(self._model.size.num_worlds,), dtype=wp.bool)
            self._base_q = wp.zeros(shape=(self._model.size.num_worlds,), dtype=wp.transformf)
            self._base_u = wp.zeros(shape=(self._model.size.num_worlds,), dtype=wp.spatial_vectorf)
            self._bodies_u_zeros = wp.zeros(shape=(self._model.size.sum_of_num_bodies,), dtype=wp.spatial_vectorf)
            self._actuators_q = wp.zeros(shape=(self._model.size.sum_of_num_actuated_joint_coords,), dtype=wp.float32)
            self._actuators_u = wp.zeros(shape=(self._model.size.sum_of_num_actuated_joint_dofs,), dtype=wp.float32)

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
        state: StateKamino,
        world_mask: wp.array[wp.bool] | None = None,
        config: SolverKamino.ResetConfig | None = None,
        success_mask: wp.array[wp.bool] | None = None,
    ):
        """
        Reset the Kamino solver state.

        Performs a configurable in-place reset of the simulation state, in all or a subset
        of worlds, setting body poses and velocities selectively to default or current values,
        or as per joint coordinates/velocities, using a forward kinematics solve.
        This is optionally combined with a reset of the pose and velocity of the floating base.

        All state components are reset consistently with the new body poses and velocities
        (unless prescribed otherwise by state flags), and solver-internal buffers are cleared.

        Args:
            state: The simulation state to reset (modified in place).
            world_mask: Optional array of per-world masks indicating which
                worlds should be reset.
                Shape of ``(num_worlds,)``.
            config: Optional reset configuration, controlling the reset behavior
                for body poses/velocities as well as floating base pose/velocity.
                If not provided, all components are reset to default (initial) values.
            success_mask: Optional mask, filled with a success boolean per world if provided
                (True if reset successfully, False if not reset due to world_mask, or if reset
                was unsuccessful, e.g. due to an unconverged FK solve).
        """

        def _check_length(data: wp.array[Any], name: str, expected: int):
            if data is not None and data.shape[0] != expected:
                raise ValueError(f"Invalid shape for {name}: Expected ({expected},), but got {data.shape}.")

        # Resolve and validate world mask
        world_mask = self._all_worlds_mask if world_mask is None else world_mask
        _check_length(world_mask, "world_mask", self._model.size.num_worlds)

        # Resolve and validate reset config
        config = SolverKamino.ResetConfig.to_default() if config is None else config
        if isinstance(config.body_poses, SolverKamino.ResetConfig.FromJointQ):
            _check_length(
                config.body_poses.joint_q,
                "config.body_poses.joint_q",
                self._model.size.sum_of_num_joint_coords,
            )
        if isinstance(config.body_poses, SolverKamino.ResetConfig.FromActuatorQ):
            _check_length(
                config.body_poses.actuator_q,
                "config.body_poses.actuator_q",
                self._model.size.sum_of_num_fk_actuated_joint_coords,
            )
        if isinstance(config.body_velocities, SolverKamino.ResetConfig.FromJointU):
            _check_length(
                config.body_velocities.joint_u,
                "config.body_velocities.joint_u",
                self._model.size.sum_of_num_joint_dofs,
            )
        if isinstance(config.body_velocities, SolverKamino.ResetConfig.FromActuatorU):
            _check_length(
                config.body_velocities.actuator_u,
                "config.body_velocities.actuator_u",
                self._model.size.sum_of_num_fk_actuated_joint_dofs,
            )
        if isinstance(config.base_pose, SolverKamino.ResetConfig.FromJointQ):
            _check_length(
                config.base_pose.joint_q,
                "config.base_pose.joint_q",
                self._model.size.sum_of_num_joint_coords,
            )
        if isinstance(config.base_pose, SolverKamino.ResetConfig.FromBaseQ):
            _check_length(config.base_pose.base_q, "config.base_pose.base_q", self._model.size.num_worlds)
        if isinstance(config.base_velocity, SolverKamino.ResetConfig.FromJointU):
            _check_length(
                config.base_velocity.joint_u,
                "config.base_velocity.joint_u",
                self._model.size.sum_of_num_joint_dofs,
            )
        if isinstance(config.base_velocity, SolverKamino.ResetConfig.FromBaseU):
            _check_length(config.base_velocity.base_u, "config.base_velocity.base_u", self._model.size.num_worlds)

        # Run the pre-reset callback if it has been set
        self._run_pre_reset_callback(state_out=state)

        # Resolve target joint_q
        joint_q = None
        if isinstance(config.body_poses, SolverKamino.ResetConfig.FromJointQ):
            joint_q = config.body_poses.joint_q
            joint_q = state.q_j if joint_q is None else joint_q

        # Resolve target joint_u
        joint_u = None
        if isinstance(config.body_velocities, SolverKamino.ResetConfig.FromJointU):
            # Reset config explicitly provides joint_u
            joint_u = config.body_velocities.joint_u
            joint_u = state.dq_j if joint_u is None else joint_u
        elif not isinstance(config.body_poses, SolverKamino.ResetConfig.Preserve) and isinstance(
            config.body_velocities, SolverKamino.ResetConfig.Preserve
        ):
            # Preserve velocities but not poses: transfer current joint_u to new poses
            joint_u = state.dq_j

        # Resolve target actuator_q and actuator_u
        actuator_q = None
        actuator_u = None
        if isinstance(config.body_poses, SolverKamino.ResetConfig.FromActuatorQ):
            actuator_q = config.body_poses.actuator_q
        if isinstance(config.body_velocities, SolverKamino.ResetConfig.FromActuatorU):
            actuator_u = config.body_velocities.actuator_u
        if joint_q is not None or joint_u is not None:
            # Extract joint state into pre-allocated actuator state buffers
            extract_actuators_state_from_joints(
                model=self._model,
                world_mask=world_mask,
                joint_q=joint_q if joint_q is not None else state.q_j,
                joint_u=joint_u if joint_u is not None else state.dq_j,
                actuator_q=self._actuators_q,
                actuator_u=self._actuators_u,
            )
            actuator_q = self._actuators_q if joint_q is not None else actuator_q
            actuator_u = self._actuators_u if joint_u is not None else actuator_u

        # Resolve target base_q
        base_q = None
        if isinstance(config.base_pose, SolverKamino.ResetConfig.ToDefault):
            # Set base pose to default if body poses are not already reset to default
            if not isinstance(config.body_poses, SolverKamino.ResetConfig.ToDefault):
                get_base_q_from_joint_q_and_body_q(
                    model=self._model,
                    joint_q=self._model.joints.q_j_0,
                    body_q=self._model.bodies.q_i_0,
                    base_q=self._base_q,
                    world_mask=world_mask,
                )
                base_q = self._base_q
        elif isinstance(config.base_pose, SolverKamino.ResetConfig.Preserve):
            # Extract current base pose if body poses are modified but base should be preserved
            if not isinstance(config.body_poses, SolverKamino.ResetConfig.Preserve):
                get_base_q_from_joint_q_and_body_q(
                    model=self._model, joint_q=state.q_j, body_q=state.q_i, base_q=self._base_q, world_mask=world_mask
                )
                base_q = self._base_q
        elif isinstance(config.base_pose, SolverKamino.ResetConfig.FromJointQ):
            # Extract base pose from provided joint_q (defaulting to extraction from body_q_0 if no base joint)
            joint_q = config.base_pose.joint_q
            joint_q = state.q_j if joint_q is None else joint_q
            get_base_q_from_joint_q_and_body_q(
                model=self._model,
                joint_q=joint_q,
                body_q=self._model.bodies.q_i_0,
                base_q=self._base_q,
                world_mask=world_mask,
            )
            base_q = self._base_q
        elif isinstance(config.base_pose, SolverKamino.ResetConfig.FromBaseQ):
            # Set base_q to provided value
            base_q = config.base_pose.base_q

        # Resolve target base_u
        base_u = None
        relative_base_u = False
        if isinstance(config.base_velocity, SolverKamino.ResetConfig.ToDefault):
            # Set base velocity to zero if body velocities are not already reset to zero
            if not isinstance(config.body_velocities, SolverKamino.ResetConfig.ToDefault):
                self._base_u.zero_()
                base_u = self._base_u
        elif isinstance(config.base_velocity, SolverKamino.ResetConfig.Preserve):
            # Extract current base velocity if body poses/velocities are modified but base velocity should be preserved
            if not isinstance(config.body_poses, SolverKamino.ResetConfig.Preserve) or not isinstance(
                config.body_velocities, SolverKamino.ResetConfig.Preserve
            ):
                get_base_u_from_joint_u_and_body_u(
                    model=self._model, joint_u=state.dq_j, body_u=state.u_i, base_u=self._base_u, world_mask=world_mask
                )
                base_u = self._base_u
                relative_base_u = True  # We preserve base_u relative to the transform applied due to base_q
        elif isinstance(config.base_velocity, SolverKamino.ResetConfig.FromJointU):
            # Extract base velocity from provided joint_u (defaulting to extraction from body_u_0 if no base joint)
            joint_u = config.base_velocity.joint_u
            joint_u = state.dq_j if joint_u is None else joint_u
            get_base_u_from_joint_u_and_body_u(
                model=self._model,
                joint_u=joint_u,
                body_u=self._model.bodies.u_i_0,
                base_u=self._base_u,
                world_mask=world_mask,
            )
            base_u = self._base_u
        elif isinstance(config.base_velocity, SolverKamino.ResetConfig.FromBaseU):
            # Set base_u to provided value
            base_u = config.base_velocity.base_u

        # Body poses: run FK or reset to default if applicable
        if actuator_q is not None:
            if self._solver_fk is None:
                raise RuntimeError("The FK solver must be enabled to use resets from joint coordinates.")
            self._solver_fk.run_fk_solve(
                actuators_q=actuator_q,
                bodies_q=state.q_i,
                base_q=base_q,
                actuators_u=actuator_u,
                base_u=base_u if actuator_u is not None else None,
                bodies_u=state.u_i if actuator_u is not None else None,
                world_mask=world_mask,
            )
        elif isinstance(config.body_poses, SolverKamino.ResetConfig.ToDefault):
            set_body_q(
                model=self._model, body_q_in=self._model.bodies.q_i_0, body_q_out=state.q_i, world_mask=world_mask
            )

        # Body velocities: run FK (if not already done) or reset to default if needed
        if actuator_u is not None and actuator_q is None:  # Velocity-level only FK
            if self._solver_fk is None:
                raise RuntimeError("The FK solver must be enabled to use resets from joint velocities.")
            self._solver_fk.solve_for_body_velocities(
                actuators_u=actuator_u,
                bodies_q=state.q_i,
                bodies_u=state.u_i,
                base_u=base_u,
                target_rel_transforms=None,
                world_mask=world_mask,
            )
        elif isinstance(config.body_velocities, SolverKamino.ResetConfig.ToDefault):
            reset_body_velocities(self._model, state, world_mask)

        # Base pose and velocity: transform body poses and velocities if not already passed to FK
        apply_base_q_needed = base_q is not None and actuator_q is None
        apply_base_u_needed = base_u is not None and actuator_u is None
        if apply_base_q_needed or apply_base_u_needed:
            set_floating_base(
                model=self._model,
                base_q=base_q if apply_base_q_needed else None,
                base_u=base_u if apply_base_u_needed else None,
                body_q=state.q_i,
                body_u=state.u_i,
                world_mask=world_mask,
                relative_base_u=relative_base_u,
            )

        # Fill/reset remaining state components based on body poses and velocities
        reset_joints_state_from_bodies_state(self._model, state, world_mask)
        reset_body_wrenches(self._model, state, world_mask)

        # Reset solver internals
        self._reset_solver_data(world_mask=world_mask)

        # Fill success mask
        if success_mask is not None:
            # Currently, only the position-level (iterative) FK solve can fail
            if actuator_q is not None:
                wp.copy(success_mask, self._solver_fk.newton_success)
            else:
                wp.copy(success_mask, world_mask)

        # Run the post-reset callback if it has been set
        self._run_post_reset_callback(state_out=state)

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
            state_in: The input current state of the simulation.
            state_out: The output next state after time integration.
            control: The input controls applied to the system.
            contacts: The set of active contacts.
            detector: An optional collision detector to use for generating contacts at the current state.
                If `None`, the `contacts` data will be used as the current set of active contacts.
            dt: A uniform time-step to apply uniformly to all worlds of the simulation.
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
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
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

    def _reset_solver_data(self, world_mask: wp.array[wp.bool] | None = None):
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

        # Reset the warm-starting caches if enabled
        if self._warmstart_mode == PADMMWarmStartMode.CONTAINERS:
            self._ws_limits.reset(world_mask=world_mask)
            self._ws_contacts.reset(world_mask=world_mask)

    ###
    # Internals - Step Operations
    ###

    def _update_joints_data(self, q_j_p: wp.array[wp.float32] | None = None):
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
            state_in: State of the system at the current time-step.
            state_out: State of the system at the next time-step.
            control: Input controls applied to the system.
            limits: Optional container for joint limits.
                If `None`, joint limit handling is skipped.
            contacts: Optional container of active contacts.
                If `None`, the solver will use the internal collision detector
                if the model admits contacts, or skip contact handling if not.
            detector: Optional collision detector.
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
