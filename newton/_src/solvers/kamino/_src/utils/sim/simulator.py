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

from ...core.builder import ModelBuilderKamino
from ...core.control import ControlKamino
from ...core.model import ModelKamino
from ...core.state import StateKamino
from ...core.types import FloatArrayLike
from ...geometry import CollisionDetector
from ...solver_kamino_impl import SolverKaminoImpl

###
# Module interface
###

__all__ = [
    "Simulator",
    "SimulatorData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class SimulatorData:
    """
    Holds the time-varying data for the simulation.

    Attributes:
        state_p (StateKamino):
            The previous state data of the simulation
        state_n (StateKamino):
            The current state data of the simulation, computed from the previous step as:
            ``state_n = f(state_p, control)``, where ``f()`` is the system dynamics function.
        control (ControlKamino):
            The control data, computed at each step as:
            ``control = g(state_n, state_p, control)``, where ``g()`` is the control function.
    """

    def __init__(self, model: ModelKamino):
        """
        Initializes the simulator data for the given model on the specified device.
        """
        self.state_p: StateKamino = model.state(device=model.device)
        self.state_n: StateKamino = model.state(device=model.device)
        self.control: ControlKamino = model.control(device=model.device)

    def cache_state(self):
        """
        Updates the previous-step caches of the state and control data from the next-step.
        """
        self.state_p.copy_from(self.state_n)


###
# Interfaces
###


class Simulator:
    """
    A high-level interface for executing physics simulations using Kamino.

    The Simulator class encapsulates the entire simulation pipeline, including model definition,
    state management, collision detection, constraint handling, and time integration.

    A Simulator is typically instantiated from a :class:`ModelBuilderKamino` that defines the model
    to be simulated. The simulator manages the time-stepping loop, invoking callbacks at various
    stages of the simulation step, and provides access to the current state and control inputs.

    Example:
    ```python
        # Create a model builder and define the model
        builder = ModelBuilderKamino()

        # Define the model components (e.g., bodies, joints, collision geometries etc.)
        builder.add_rigid_body(...)
        builder.add_joint(...)
        builder.add_geometry(...)

        # Create the simulator from the builder
        simulator = Simulator(builder)

        # Run the simulation for a specified number of steps
        for _i in range(num_steps):
            simulator.step()
    ```
    """

    @dataclass
    class Config:
        """
        Holds the configuration for the simulator.
        """

        dt: float | FloatArrayLike = 0.001
        """
        The time-step to be used for the simulation.\n
        Defaults to `0.001` seconds.
        """

        collision_detector: CollisionDetector.Config = field(default_factory=CollisionDetector.Config)
        """
        The config for the collision detector.
        See :class:`CollisionDetector.Config` for more details.
        """

        solver: SolverKaminoImpl.Config = field(default_factory=SolverKaminoImpl.Config)
        """
        The config for the dynamics solver.\n
        See :class:`SolverKaminoImpl.Config` for more details.
        """

        def validate(self) -> None:
            """
            Validates the simulator configurations.
            """
            # First check the time-step
            if isinstance(self.dt, float):
                if self.dt != self.dt:
                    raise ValueError("Invalid time-step: cannot be NaN.")
                if self.dt <= 0.0:
                    raise ValueError(f"Invalid time-step: got {self.dt}, but must be a positive value.")
            elif isinstance(self.dt, FloatArrayLike):
                if len(self.dt) == 0:
                    raise ValueError("Invalid time-step array: cannot be empty.")
                elif any(dt <= 0.0 or dt != dt for dt in self.dt):
                    raise ValueError("Invalid time-step array: all values must be positive and non-NaN.")
                elif not all(isinstance(dt, float) for dt in self.dt):
                    raise TypeError("Invalid time-step array: all values must be of type float.")
            else:
                raise TypeError("Invalid time-step: must be a `float` or a `FloatArrayLike`.`")

            # Ensure nested configs are properly created
            if not isinstance(self.collision_detector, CollisionDetector.Config):
                raise TypeError(f"Invalid type for collision_detector config: {type(self.collision_detector)}")
            if not isinstance(self.solver, SolverKaminoImpl.Config):
                raise TypeError(f"Invalid type for solver config: {type(self.solver)}")

            # Then check the nested config values
            self.collision_detector.validate()
            self.solver.validate()

        def __post_init__(self):
            """Post-initialization processing to ensure nested configs are properly created."""
            self.validate()

    SimCallbackType = Callable[["Simulator"], None]
    """Defines a common type signature for all simulator callback functions."""

    def __init__(
        self,
        builder: ModelBuilderKamino,
        config: Simulator.Config = None,
        device: wp.DeviceLike = None,
    ):
        """
        Initializes the simulator with the given model builder, time-step, and device.

        Args:
            builder (ModelBuilderKamino): The model builder defining the model to be simulated.
            config (Simulator.Config, optional): The simulator config to use. If None, the default config are used.
            device (wp.DeviceLike, optional): The device to run the simulation on. If None, the default device is used.
        """
        # Cache simulator config: If no config is provided, use default configs
        if config is None:
            config = Simulator.Config()
        config.validate()
        self._config: Simulator.Config = config

        # Cache the target device use for the simulation
        self._device: wp.DeviceLike = device

        # Pass collision detector config to builder before finalization
        if self._config.collision_detector.max_contacts_per_pair is not None:
            builder.max_contacts_per_pair = self._config.collision_detector.max_contacts_per_pair

        # Finalize the model from the builder on the specified
        # device, allocating all necessary model data structures
        self._model = builder.finalize(device=self._device)

        # Configure model time-steps across all worlds
        if isinstance(self._config.dt, float):
            self._model.time.set_uniform_timestep(self._config.dt)
        elif isinstance(self._config.dt, FloatArrayLike):
            self._model.time.set_timesteps(self._config.dt)

        # Allocate time-varying simulation data
        self._data = SimulatorData(model=self._model)

        # Allocate collision detection and contacts interface
        self._collision_detector = CollisionDetector(
            model=self._model,
            config=self._config.collision_detector,
        )

        # Capture a reference to the contacts manager
        self._contacts = self._collision_detector.contacts

        # Define a physics solver for time-stepping
        self._solver = SolverKaminoImpl(
            model=self._model,
            contacts=self._contacts,
            config=self._config.solver,
        )

        # Initialize callbacks
        self._pre_reset_cb: Simulator.SimCallbackType = None
        self._post_reset_cb: Simulator.SimCallbackType = None
        self._control_cb: Simulator.SimCallbackType = None

        # Initialize the simulation state
        with wp.ScopedDevice(self._device):
            self.reset()

    ###
    # Properties
    ###

    @property
    def config(self) -> Simulator.Config:
        """
        Returns the simulator config.
        """
        return self._config

    @property
    def model(self) -> ModelKamino:
        """
        Returns the time-invariant simulation model data.
        """
        return self._model

    @property
    def data(self) -> SimulatorData:
        """
        Returns the simulation data container.
        """
        return self._data

    @property
    def state(self) -> StateKamino:
        """
        Returns the current state of the simulation.
        """
        return self._data.state_n

    @property
    def state_previous(self) -> StateKamino:
        """
        Returns the previous state of the simulation.
        """
        return self._data.state_p

    @property
    def control(self) -> ControlKamino:
        """
        Returns the current control inputs of the simulation.
        """
        return self._data.control

    @property
    def limits(self):
        """
        Returns the limits container of the simulation.
        """
        return self._solver._limits

    @property
    def contacts(self):
        """
        Returns the contacts container of the simulation.
        """
        return self._contacts

    @property
    def metrics(self):
        """
        Returns the current simulation metrics.
        """
        return self._solver.metrics

    @property
    def collision_detector(self) -> CollisionDetector:
        """
        Returns the collision detector.
        """
        return self._collision_detector

    @property
    def solver(self) -> SolverKaminoImpl:
        """
        Returns the physics step solver.
        """
        return self._solver

    ###
    # Configurations - Callbacks
    ###

    def set_pre_reset_callback(self, callback: SimCallbackType):
        """
        Sets a reset callback to be called at the beginning of each call to `reset_*()` methods.
        """
        self._pre_reset_cb = callback

    def set_post_reset_callback(self, callback: SimCallbackType):
        """
        Sets a reset callback to be called at the end of each call to to `reset_*()` methods.
        """
        self._post_reset_cb = callback

    def set_control_callback(self, callback: SimCallbackType):
        """
        Sets a control callback to be called at the beginning of the step, that
        should populate `data.control`, i.e. the control inputs for the current
        step, based on the current and previous states and controls.
        """
        self._control_cb = callback

    ###
    # Operations
    ###

    def reset(
        self,
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
        which worlds should be reset.

        Args:
            world_mask (wp.array, optional):
                Optional array of per-world masks indicating which worlds should be reset.
                Shape of `(num_worlds,)` and type :class:`wp.int8 | wp.bool`
            joint_q (wp.array, optional):
                Optional array of target joint coordinates.
                Shape of `(num_joint_coords,)` and type :class:`wp.float32`
            joint_qd (wp.array, optional):
                Optional array of target joint DoF velocities.
                Shape of `(num_joint_dofs,)` and type :class:`wp.float32`
            base_q (wp.array, optional):
                Optional array of target base body poses.
                Shape of `(num_worlds,)` and type :class:`wp.transformf`
            base_qd (wp.array, optional):
                Optional array of target base body twists.
                Shape of `(num_worlds,)` and type :class:`wp.spatial_vectorf`
        """
        # Run the pre-reset callback if it has been set
        if self._pre_reset_cb is not None:
            self._pre_reset_cb(self)

        # Step the physics solver
        self._solver.reset(
            state_out=self._data.state_n,
            world_mask=world_mask,
            actuator_q=actuator_q,
            actuator_u=actuator_u,
            joint_q=joint_q,
            joint_u=joint_u,
            base_q=base_q,
            base_u=base_u,
            bodies_q=bodies_q,
            bodies_u=bodies_u,
        )

        # Cache the current state as the previous state for the next step
        self._data.cache_state()

        # Run the post-reset callback if it has been set
        if self._post_reset_cb is not None:
            self._post_reset_cb(self)

    def step(self):
        """
        Advances the simulation by a single time-step.
        """
        # Run the control callback if it has been set
        if self._control_cb is not None:
            self._control_cb(self)

        # Cache the current state as the previous state for the next step
        self._data.cache_state()

        # Step the physics solver
        self._solver.step(
            state_in=self._data.state_p,
            state_out=self._data.state_n,
            control=self._data.control,
            contacts=self._contacts,
            detector=self._collision_detector,
        )
