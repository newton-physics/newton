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

"""Provides a high-level interface for physics simulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import warp as wp
from warp.context import Devicelike

from ..core.bodies import update_body_inertias, update_body_wrenches
from ..core.builder import ModelBuilder
from ..core.control import Control
from ..core.model import Model, ModelData
from ..core.state import State
from ..core.time import advance_time
from ..dynamics.dual import DualProblem, DualProblemSettings
from ..dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from ..geometry.contacts import Contacts
from ..geometry.detector import CollisionDetector
from ..integrators.euler import integrate_semi_implicit_euler
from ..kinematics.constraints import make_unilateral_constraints_info, update_constraints_info
from ..kinematics.jacobians import DenseSystemJacobians
from ..kinematics.joints import compute_joints_state
from ..kinematics.limits import Limits
from ..linalg import LinearSolver, LLTBlockedSolver
from ..solvers.padmm import PADMMSettings, PADMMSolver

###
# Module interface
###

__all__ = [
    "Simulator",
    "SimulatorData",
    "SimulatorSettings",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class SimulatorSettings:
    """
    Holds the configuration settings for the simulator.
    """

    dt: float = 0.001
    """The time-step to be used for the simulation."""

    problem: DualProblemSettings = field(default_factory=DualProblemSettings)
    """The settings for the dynamics problem."""

    solver: PADMMSettings = field(default_factory=PADMMSettings)
    """The settings for the dynamics solver."""

    use_solver_acceleration: bool = True
    """Set to True to enable Nesterov-type acceleration, i.e. use APADMM instead of standard PADMM."""

    collect_solver_info: bool = False
    """Set to True to collect solver convergence and performance info at each simulation step."""

    linear_solver_type: type[LinearSolver] = LLTBlockedSolver
    """The type of linear solver to use for the dynamics problem."""

    def check(self) -> None:
        """
        Checks the validity of the settings.
        """
        if self.dt <= 0.0:
            raise ValueError(f"Invalid time-step: {self.dt}. Must be a positive value.")
        self.problem.check()
        self.solver.check()


class SimulatorData:
    """
    Holds the time-varying data for the simulation.

    Attributes:
        state (ModelData): holds the internal solver state
        s_p (State): holds the 'previous' state data
        c_p (Control): holds the 'previous' control data
        s_n (State): holds the 'current' state data, computed from the previous step as:
            ``s_n = f(s_p, c_p)``, where ``f()`` is the system dynamics function.
        c_n (Control): holds the 'current' control data, computed at each step as:
            ``c_n = g(s_n, s_p, c_p)``, where ``g()`` is the control policy function.
    """

    def __init__(self, model: Model, device: Devicelike = None):
        """
        Initializes the simulator data for the given model on the specified device.
        """
        # First allocate the compact state and control containers for the previous and next steps
        # NOTE: The `next` state is to be understood as the current state, previous is always the past
        self.state_n: State = model.state(device=device)
        self.state_p: State = model.state(device=device)
        self.control_n: Control = model.control(device=device)
        self.control_p: Control = model.control(device=device)

        # Then allocate the internal solver state container
        self.solver: ModelData = model.data(device=device)

    def update_previous(self):
        """
        Updates the previous-step caches of the state and control data from the next-step.
        """
        self.state_p.copy_from(self.state_n)
        self.control_p.copy_from(self.control_n)

    def update_next(self):
        """
        Synchronizes the next state with the internal solver state data.

        Note:
        This is necessary since the integrator updates the next state in-place,
        while all joint and body wrenches attributes are updated by the solver.
        """
        wp.copy(self.state_n.q_i, self.solver.bodies.q_i)
        wp.copy(self.state_n.u_i, self.solver.bodies.u_i)
        wp.copy(self.state_n.w_i, self.solver.bodies.w_i)
        wp.copy(self.state_n.q_j, self.solver.joints.q_j)
        wp.copy(self.state_n.dq_j, self.solver.joints.dq_j)
        wp.copy(self.state_n.lambda_j, self.solver.joints.lambda_j)


###
# Interfaces
###


class Simulator:
    """
    A high-level interface for executing physics simulations using Kamino.

    The Simulator class encapsulates the entire simulation pipeline, including model definition,
    state management, collision detection, constraint handling, and time integration.

    A Simulator is typically instantiated from a :class:`ModelBuilder` that defines the model
    to be simulated. The simulator manages the time-stepping loop, invoking callbacks at various
    stages of the simulation step, and provides access to the current state and control inputs.

    Example:
    ```python
        # Create a model builder and define the model
        builder = ModelBuilder()

        # Define the model components (e.g., bodies, joints, collision geometries etc.)
        builder.add_body(...)
        builder.add_joint(...)
        builder.add_collision_geometry(...)

        # Create the simulator from the builder
        simulator = Simulator(builder)

        # Run the simulation for a specified number of steps
        for _i in range(num_steps):
            simulator.step()
    ```
    """

    def __init__(
        self, builder: ModelBuilder, settings: SimulatorSettings = None, device: Devicelike = None, shadow: bool = False
    ):
        """
        Initializes the simulator with the given model builder, time-step, and device.
        """

        # Use default settings if none are provided
        if settings is None:
            settings = SimulatorSettings()

        # Validate the settings
        settings.check()

        # Host-side time-keeping
        self._time: float = 0.0
        self._max_time: float = 0.0
        self._steps: int = 0
        self._max_steps: int = 0

        # Cache the solver settings
        self._settings: SimulatorSettings = settings

        # Cache the target device use for the simulation
        self._device: Devicelike = device

        # Joint Limits
        self._limits = Limits(builder=builder, device=self._device)

        # Collision Detection
        self._collision_detector = CollisionDetector(builder=builder, device=self._device)

        # Model
        self._model = builder.finalize(device=self._device)

        # Configure model time-steps
        self._model.time.set_uniform_timestep(self._settings.dt)

        # Allocate system data on the device
        self._data = SimulatorData(model=self._model, device=self._device)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(
            model=self._model, data=self._data.solver, limits=self._limits, contacts=self.contacts, device=self._device
        )

        # Allocate Jacobians data on the device
        self._jacobians = DenseSystemJacobians(
            model=self._model,
            limits=self._limits,
            contacts=self._collision_detector.contacts,
            device=self._device,
        )

        # Allocate the dual problem data on the device
        self._dual_problem = DualProblem(
            model=self._model,
            data=self._data.solver,
            limits=self._limits,
            contacts=self._collision_detector.contacts,
            solver=settings.linear_solver_type,
            settings=settings.problem,
            device=self._device,
        )

        # Allocate the dual solver data on the device
        self._dual_solver = PADMMSolver(
            model=self._model,
            limits=self._limits,
            contacts=self._collision_detector.contacts,
            settings=settings.solver,
            use_acceleration=settings.use_solver_acceleration,
            collect_info=settings.collect_solver_info,
            device=self._device,
        )

        # Initialize callbacks
        self._reset_cb: Callable[[Simulator], None] = None
        self._control_cb: Callable[[Simulator], None] = None
        self._pre_step_cb: Callable[[Simulator], None] = None
        self._mid_step_cb: Callable[[Simulator], None] = None
        self._post_step_cb: Callable[[Simulator], None] = None

        # Define optional data shadowing  on the CPU
        self._host: SimulatorData | None = None
        if shadow:
            self.sync_host()

        # Initialize the simulation state
        with wp.ScopedDevice(self._device):
            self.reset()

    ###
    # Properties
    ###

    @property
    def time(self) -> float:
        return self._time

    @property
    def max_time(self) -> float:
        return self._max_time

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def dt(self) -> float:
        return self._settings.dt

    @property
    def model(self) -> Model:
        return self._model

    @property
    def data(self) -> SimulatorData:
        return self._data

    @property
    def model_data(self) -> ModelData:
        return self._data.solver

    @property
    def state_previous(self) -> State:
        return self._data.state_p

    @property
    def state(self) -> State:
        return self._data.state_n

    @property
    def control_previous(self) -> Control:
        return self._data.control_p

    @property
    def control(self) -> Control:
        return self._data.control_n

    @property
    def limits(self) -> Limits:
        return self._limits

    @property
    def contacts(self) -> Contacts:
        return self._collision_detector.contacts

    @property
    def collision_detector(self) -> CollisionDetector:
        return self._collision_detector

    @property
    def jacobians(self) -> DenseSystemJacobians:
        return self._jacobians

    @property
    def problem(self) -> DualProblem:
        return self._dual_problem

    @property
    def solver(self) -> PADMMSolver:
        return self._dual_solver

    @property
    def host(self) -> SimulatorData | None:
        # return self._host
        return self._data

    ###
    # Callbacks
    ###

    def set_reset_callback(self, callback: Callable[[Simulator], None]):
        """
        Set a reset callback to be called at each call to `reset()`, that
        should populate `data.c_n`, i.e. the control inputs for the current step,
        based on the current and previous states and controls.
        """
        self._reset_cb = callback

    def set_control_callback(self, callback: Callable[[Simulator], None]):
        """
        Set a control callback to be called at the beginning of the step, that
        should populate `data.c_n`, i.e. the control inputs for the current step,
        based on the current and previous states and controls.
        """
        self._control_cb = callback

    def set_pre_step_callback(self, callback: Callable[[Simulator], None]):
        """
        Set a callback to be called before forward dynamics solve.
        """
        self._pre_step_cb = callback

    def set_mid_step_callback(self, callback: Callable[[Simulator], None]):
        """
        Set a callback to be called between forward dynamics solver and state integration.
        """
        self._mid_step_cb = callback

    def set_post_step_callback(self, callback: Callable[[Simulator], None]):
        """
        Set a callback to be called after state integration.
        """
        self._post_step_cb = callback

    ###
    # Internals
    ###

    def _reset_time(self):
        """
        Resets the time and step count of the simulation.
        """
        self._time = 0.0
        self._steps = 0
        self._data.solver.time.zero()

    def _reset_bodies(self):
        """
        Resets the state of all bodies to the initial states defined in the model.
        """
        # First set the active body states to the initial states defined in the model
        wp.copy(self._data.solver.bodies.q_i, self._model.bodies.q_i_0)
        wp.copy(self._data.solver.bodies.u_i, self._model.bodies.u_i_0)

        # Then update the in-world-frame body inertias from the body states
        update_body_inertias(model=self._model.bodies, data=self._data.solver.bodies)

        # Finally, clear all body wrenches by setting them to zero
        self._data.solver.bodies.clear_all_wrenches()

    def _reset_joints(self):
        """
        Resets the state of all joints according to the initial state of the bodies.
        """
        # First clear all joint states (i.e. generalized coordinates and velocities) to zeros
        # NOTE: We do this so that the previous state is always zeroed out on reset. This is
        # necessary as the `compute_joints_state()` operation will use the previous joint state
        # to detect roll-over for rotational coordinates/DoFs.
        self._data.solver.joints.clear_state()

        # Then compute the initial joint states based on the body states
        compute_joints_state(model=self._model, q_j_p=self._data.solver.joints.q_j, data=self._data.solver)

        # Finally, clear all joint constraint reactions,
        # actuation forces, and wrenches, setting them to zero
        self._data.solver.joints.clear_constraint_reactions()
        self._data.solver.joints.clear_actuation_forces()
        self._data.solver.joints.clear_wrenches()

    def _reset_states_and_controls(self):
        """
        Resets all state and control data to match the internal solver state.
        """
        # First clear the next-step control inputs so they correctly propagate to the previous-step
        self._data.control_n.tau_j.zero_()

        # Then update the next-step state from the internal solver state
        self._data.update_next()

        # Finally, update the previous-step state and control from the next-step values
        self._data.update_previous()

    def _run_reset_callback(self):
        """
        Run the reset callback if it has been set.
        """
        if self._reset_cb is not None:
            self._reset_cb(self)

    def _run_control_callback(self):
        """
        Run the control callback if it has been set.
        """
        if self._control_cb is not None:
            self._control_cb(self)
            wp.copy(self._data.solver.joints.tau_j, self._data.control_n.tau_j)

    def _run_prestep_callback(self):
        """
        Run the pre-step callback if it has been set.
        """
        if self._pre_step_cb is not None:
            self._pre_step_cb(self)

    def _run_midstep_callback(self):
        """
        Run the mid-step callback if it has been set.
        """
        if self._mid_step_cb is not None:
            self._mid_step_cb(self)

    def _run_poststep_callback(self):
        """
        Executes the post-step callback if it has been set.
        """
        if self._post_step_cb is not None:
            self._post_step_cb(self)

    def _update_actuation_wrenches(self):
        """
        Updates the actuation wrenches based on the current control inputs.
        """
        compute_joint_dof_body_wrenches(self._model, self._data.solver, self._jacobians.data)

    def _check_limits(self):
        """
        Runs limit detection to generate active joint limits.
        """
        self._limits.detect(self._model, self._data.solver)

    def _collide(self):
        """
        Runs collision detection to generate for active contacts.
        """
        self._collision_detector.collide(self._model, self._data.solver)

    def _update_constraint_info(self):
        """
        Updates the state info with the set of active constraints resulting from limit and collision detection.
        """
        update_constraints_info(model=self._model, data=self._data.solver)

    def _forward_intermediate(self):
        """
        Updates intermediate quantities required for the forward dynamics solve.
        """
        update_body_inertias(model=self._model.bodies, data=self._data.solver.bodies)

    def _forward_kinematics(self):
        """
        Updates the forward kinematics by building the system Jacobians (of actuation and
        constraints) based on the current state of the system and set of active constraints.
        """
        self._jacobians.build(
            model=self._model,
            data=self._data.solver,
            limits=self._limits.data,
            contacts=self.contacts.data,
            reset_to_zero=True,
        )

    def _forward_dynamics(self):
        """
        Constructs the forward dynamics problem quantities based on the current state of
        the system, the set of active constraints, and the updated system Jacobians.
        """
        self._dual_problem.build(
            model=self._model,
            data=self._data.solver,
            limits=self._limits.data,
            contacts=self.contacts.data,
            jacobians=self.jacobians.data,
            reset_to_zero=True,
        )

    def _forward_constraints(self):
        """
        Solves the forward dynamics sub-problem to compute constraint
        reactions and body wrenches effected through constraints.
        """
        # Solve the dual problem to compute the constraint reactions
        self._dual_solver.solve(problem=self._dual_problem)

        # Compute the effective body wrenches applied by the set of
        # active constraints from the respective reaction multipliers
        compute_constraint_body_wrenches(
            model=self._model,
            data=self._data.solver,
            limits=self._limits.data,
            contacts=self.contacts.data,
            jacobians=self._jacobians.data,
            lambdas_offsets=self._dual_problem.data.vio,
            lambdas_data=self._dual_solver.data.solution.lambdas,
        )

    def _forward_wrenches(self):
        """
        Computes the total (i.e. net) body wrenches by summing up all individual contributions,
        from joint actuation, joint limits, contacts, and purely external effects.
        """
        update_body_wrenches(self._model.bodies, self._data.solver.bodies)

    def _forward(self):
        """
        Solves the forward dynamics sub-problem to compute constraint reactions
        and total effective body wrenches applied to each body of the system.
        """
        # # Update intermediate quantities
        self._forward_intermediate()

        # Update the kinematics
        self._forward_kinematics()

        # Update the dynamics
        self._forward_dynamics()

        # Compute constraint reactions
        self._forward_constraints()

        # Post-processing
        self._forward_wrenches()

    def _integrate(self):
        """
        Solves the time integration sub-problem to compute the next state of the system.
        """

        # Update the caches of the previous-step state and control data from the updated next-step
        # NOTE: This needs to happen before the time-integrator updates the next-state in-place
        self._data.update_previous()

        # Integrate the state of the system (i.e. of the bodies) to compute the next state
        integrate_semi_implicit_euler(model=self._model, data=self._data.solver)

        # Update the joint states based on the updated body states
        # NOTE: We use the previous state `state_p` for post-processing
        # purposes, e.g. account for roll-over of revolute joints etc
        compute_joints_state(model=self._model, q_j_p=self._data.state_p.q_j, data=self._data.solver)

        # Update the next-step state from the internal solver state
        self._data.update_next()

    def _advance_time(self):
        """
        Updates simulation time-keeping (i.e. physical time and discrete steps).
        """
        self._steps += 1
        self._time += self._settings.dt
        advance_time(self._model.time, self._data.solver.time)

    ###
    # Front-end Operations
    ###

    def reset(self):
        """
        Resets the simulation to the initial state defined in the model.
        """
        # Reset the time and step count
        self._reset_time()

        # First reset the states of all bodies
        self._reset_bodies()

        # Then reset the state of all joints
        self._reset_joints()

        # Finally, reset all state and control
        # data to match the internal solver state
        self._reset_states_and_controls()

        # Update the kinematics
        # NOTE: This constructs the system Jacobians, which ensures
        # that control action can be applied on the first call to `step()`
        self._forward_kinematics()

        # Run the reset callback if it has been set
        self._run_reset_callback()

    def step(self):
        """
        Advances the simulation by a single time-step.
        """
        # Run the control callback if it has been set
        self._run_control_callback()

        # Compute the body actuation wrenches based on the current control inputs
        self._update_actuation_wrenches()

        # Run limit detection to generate active joint limits
        self._check_limits()

        # Run collision detection to generate for active contacts
        self._collide()

        # Update the constraint state info
        self._update_constraint_info()

        # Run the pre-step callback if it has been set
        self._run_prestep_callback()

        # Solve the forward dynamics sub-problem to compute constraint reactions and body wrenches
        self._forward()

        # Run the mid-step callback if it has been set
        self._run_midstep_callback()

        # Solve the time integration sub-problem to compute the next state of the system
        self._integrate()

        # Run the post-step callback if it has been set
        self._run_poststep_callback()

        # Update time-keeping (i.e. physical time and discrete steps)
        self._advance_time()

    def sync_host(self):
        """
        Updates the host-side data with the in-device data.
        """
        # Construct the host data if it does not exist
        if self._host is None:
            self._host = SimulatorData(model=self._model, device="cpu")
        # Update the host data from the device data
        # TODO: Implement the host data update
        # self._host.solver = self._data.solver
