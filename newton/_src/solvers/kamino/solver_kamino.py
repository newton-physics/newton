# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines the :class:`SolverKamino` class, providing a physics backend for
simulating constrained multi-body systems for arbitrary mechanical assemblies.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import warp as wp

from ...core.types import override
from ...sim import (
    Contacts,
    Control,
    JointType,
    Model,
    ModelBuilder,
    State,
)
from ..flags import SolverNotifyFlags
from ..solver import SolverBase

if TYPE_CHECKING:
    from .config import (
        CollisionDetectorConfig,
        ConfigBase,
        ConstrainedDynamicsConfig,
        ConstraintStabilizationConfig,
        ForwardKinematicsSolverConfig,
        PADMMSolverConfig,
    )

###
# Module interface
###

__all__ = ["SolverKamino"]


###
# Interfaces
###


class SolverKamino(SolverBase):
    """
    A physics solver for simulating constrained multi-body systems containing kinematic loops,
    under-/overactuation, joint-limits, hard frictional contacts and restitutive impacts.

    This solver uses the Proximal-ADMM algorithm to solve the forward dynamics formulated
    as a Nonlinear Complementarity Problem (NCP) over the set of bilateral kinematic joint
    constraints and unilateral constraints that include joint-limits and contacts.

    .. note::
        Currently still in `Beta`, so we do not recommend using this solver for
        production use cases yet, as we expect many things to change in future releases.
        This includes both the public API and internal implementation; adding support for
        more simulation features (e.g. joints, constraints, actuators), performance
        optimizations, and bug fixes.

    References:
        - Tsounis, Vassilios, Ruben Grandia, and Moritz Bächer.
          On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops.
          arXiv preprint arXiv:2504.19771 (2025).
          https://doi.org/10.48550/arXiv.2504.19771
        - Carpentier, Justin, Quentin Le Lidec, and Louis Montaut.
          From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach.
          20th edition of the “Robotics: Science and Systems”(RSS) Conference. 2024.
          https://roboticsproceedings.org/rss20/p108.pdf
        - Tasora, A., Mangoni, D., Benatti, S., & Garziera, R. (2021).
          Solving variational inequalities and cone complementarity problems in
          nonsmooth dynamics using the alternating direction method of multipliers.
          International Journal for Numerical Methods in Engineering, 122(16), 4093-4113.
          https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6693

    After constructing :class:`ModelKamino`, :class:`StateKamino`, :class:`ControlKamino` and :class:`ContactsKamino`
    objects, this physics solver may be used to advance the simulation state forward in time.

    Example
    -------

    .. code-block:: python

        config = newton.solvers.SolverKamino.Config()
        solver = newton.solvers.SolverKamino(model, config)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    @dataclass
    class Config:
        """
        A container to hold all configurations of the :class:`SolverKamino` solver.
        """

        sparse_jacobian: bool = False
        """
        Flag to indicate whether the solver should use sparse data representations for the Jacobian.
        """

        sparse_dynamics: bool = False
        """
        Flag to indicate whether the solver should use sparse data representations for the dynamics.
        """

        use_collision_detector: bool = False
        """
        Flag to indicate whether the Kamino-provided collision detector should be used.
        """

        use_fk_solver: bool = False
        """
        Flag to indicate whether the Kamino-provided FK solver should be enabled.\n

        The FK solver is used for computing consistent initial states given input
        joint positions, joint velocities and optional base body poses and twists.

        It is specifically designed to handle the presence of:
        - kinematic loops
        - passive joints
        - over/under-actuation
        """

        collision_detector: CollisionDetectorConfig | None = None
        """
        Configurations for the collision detector.\n
        See :class:`CollisionDetectorConfig` for more details.\n
        If `None`, the default configuration will be used.
        """

        constraints: ConstraintStabilizationConfig | None = None
        """
        Configurations for the constraint stabilization parameters.\n
        See :class:`ConstraintStabilizationConfig` for more details.\n
        If `None`, default values will be used.
        """

        dynamics: ConstrainedDynamicsConfig | None = None
        """
        Configurations for the constrained dynamics problem.\n
        See :class:`ConstrainedDynamicsConfig` for more details.\n
        If `None`, default values will be used.
        """

        padmm: PADMMSolverConfig | None = None
        """
        Configurations for the dynamics solver.\n
        See :class:`PADMMSolverConfig` for more details.\n
        If `None`, default values will be used.
        """

        fk: ForwardKinematicsSolverConfig | None = None
        """
        Configurations for the forward kinematics solver.\n
        See :class:`ForwardKinematicsSolverConfig` for more details.\n
        If `None`, default values will be used.
        """

        rotation_correction: Literal["twopi", "continuous", "none"] = "twopi"
        """
        The rotation correction mode to use for rotational DoFs.\n
        See :class:`JointCorrectionMode` for available options.
        Defaults to `twopi`.
        """

        integrator: Literal["euler", "moreau"] = "euler"
        """
        The time-integrator to use for state integration.\n
        See available options in the `integrators` module.\n
        Defaults to `"euler"`.
        """

        angular_velocity_damping: float = 0.0
        """
        A damping factor applied to the angular velocity of bodies during state integration.\n
        This can help stabilize simulations with large time steps or high angular velocities.\n
        Defaults to `0.0` (i.e. no damping).
        """

        collect_solver_info: bool = False
        """
        Enables/disables collection of solver convergence and performance info at each simulation step.\n
        Enabling this option as it will significantly increase the runtime of the solver.\n
        Defaults to `False`.
        """

        compute_solution_metrics: bool = False
        """
        Enables/disables computation of solution metrics at each simulation step.\n
        Enabling this option as it will significantly increase the runtime of the solver.\n
        Defaults to `False`.
        """

        @staticmethod
        def register_custom_attributes(builder: ModelBuilder) -> None:
            """
            Register custom attributes for the :class:`SolverKamino.Config` configurations.

            Note: Currently, not all configurations are registered as custom attributes,
            as only those supported by the Kamino USD scene API have been included. More
            will be added in the future as latter is being developed.

            Args:
                builder: The model builder instance with which to register the custom attributes.
            """
            # Import here to avoid module-level imports and circular dependencies
            from . import config  # noqa: PLC0415
            from ._src.core.joints import JointCorrectionMode  # noqa: PLC0415

            # Register KaminoSceneAPI custom attributes for each sub-configuration container
            config.ForwardKinematicsSolverConfig.register_custom_attributes(builder)
            config.ConstraintStabilizationConfig.register_custom_attributes(builder)
            config.ConstrainedDynamicsConfig.register_custom_attributes(builder)
            config.CollisionDetectorConfig.register_custom_attributes(builder)
            config.PADMMSolverConfig.register_custom_attributes(builder)

            # Register KaminoSceneAPI custom attributes for each individual solver-level configurations
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="joint_correction",
                    frequency=Model.AttributeFrequency.ONCE,
                    assignment=Model.AttributeAssignment.MODEL,
                    dtype=str,
                    default="twopi",
                    namespace="kamino",
                    usd_attribute_name="newton:kamino:jointCorrection",
                    usd_value_transformer=JointCorrectionMode.parse_usd_attribute,
                )
            )

        @staticmethod
        def from_model(model: Model, **kwargs: dict[str, Any]) -> SolverKamino.Config:
            """
            Creates a configuration container by attempting to parse
            custom attributes from a :class:`Model` if available.

            Note: If the model was imported from USD and contains custom attributes defined
            by the KaminoSceneAPI, those attributes will be parsed and used to populate
            the configuration container. Additionally, any sub-configurations that are
            provided as keyword arguments will also be used to populate the corresponding
            sections of the configuration, allowing for a combination of model-imported
            and explicit user-provided configurations. If certain configurations are not
            provided either via the model's custom attributes or as keyword arguments,
            then default values will be used.

            Args:
                model: The Newton model from which to parse configurations.
            """
            # Import here to avoid module-level imports and circular dependencies
            from . import config  # noqa: PLC0415

            # Create a base config with default values and
            # user-provided provided kwarg overrides
            cfg = SolverKamino.Config(**kwargs)

            # Parse solver-specific attributes imported from USD
            kamino_attrs = getattr(model, "kamino", None)
            if kamino_attrs is not None:
                if hasattr(kamino_attrs, "joint_correction"):
                    cfg.rotation_correction = kamino_attrs.joint_correction[0]

            # Parse sub-configurations from the provided kwargs, if available, otherwise use defaults
            subconfigs: dict[str, ConfigBase] = {
                "collision_detector": config.CollisionDetectorConfig,
                "constraints": config.ConstraintStabilizationConfig,
                "dynamics": config.ConstrainedDynamicsConfig,
                "padmm": config.PADMMSolverConfig,
                "fk": config.ForwardKinematicsSolverConfig,
            }
            for attr_name, config_cls in subconfigs.items():
                nested_config = kwargs.get(attr_name, None)
                nested_kwargs = nested_config.__dict__ if nested_config is not None else {}
                setattr(cfg, attr_name, config_cls.from_model(model, **nested_kwargs))

            # Return the fully constructed config with sub-configurations
            # parsed from the model's custom attributes if available,
            # otherwise using defaults or provided kwargs.
            return cfg

        @override
        def validate(self) -> None:
            """
            Validates the current values held by the :class:`SolverKamino.Config` instance.
            """
            # Import here to avoid module-level imports and circular dependencies
            from ._src.core.joints import JointCorrectionMode  # noqa: PLC0415

            # Ensure that the sparsity settings are compatible with each other
            if self.sparse_dynamics and not self.sparse_jacobian:
                raise ValueError(
                    "Sparsity setting mismatch: `sparse_dynamics` solver "
                    "option requires that `sparse_jacobian` is set to `True`."
                )

            # Ensure that all mandatory configurations are not None.
            if self.constraints is None:
                raise ValueError("Constraint stabilization config cannot be None.")
            elif self.dynamics is None:
                raise ValueError("Constrained dynamics config cannot be None.")
            elif self.padmm is None:
                raise ValueError("PADMM solver config cannot be None.")

            # Validate specialized sub-configurations
            # using their own built-in validations
            if self.collision_detector is not None:
                self.collision_detector.validate()
            if self.fk is not None:
                self.fk.validate()
            self.constraints.validate()
            self.dynamics.validate()
            self.padmm.validate()

            # Conversion to JointCorrectionMode will raise an error if the input string is invalid.
            JointCorrectionMode.from_string(self.rotation_correction)

            # Ensure the integrator choice is valid
            supported_integrators = {"euler", "moreau"}
            if self.integrator not in supported_integrators:
                raise ValueError(f"Invalid integrator: {self.integrator}. Must be one of {supported_integrators}.")

            # Ensure the angular velocity damping factor is non-negative
            if self.angular_velocity_damping < 0.0 or self.angular_velocity_damping > 1.0:
                raise ValueError(
                    f"Invalid angular velocity damping factor: {self.angular_velocity_damping}. "
                    "Must be in the range [0.0, 1.0]."
                )

        @override
        def __post_init__(self):
            """
            Post-initialization to default-initialize empty configurations and validate those specified by the user.
            """
            # Import here to avoid module-level imports and circular dependencies
            from . import config  # noqa: PLC0415

            # Default-initialize any sub-configurations that were not explicitly provided by the user
            if self.collision_detector is None and self.use_collision_detector:
                self.collision_detector = config.CollisionDetectorConfig()
            if self.fk is None and self.use_fk_solver:
                self.fk = config.ForwardKinematicsSolverConfig()
            if self.constraints is None:
                self.constraints = config.ConstraintStabilizationConfig()
            if self.dynamics is None:
                self.dynamics = config.ConstrainedDynamicsConfig()
            if self.padmm is None:
                self.padmm = config.PADMMSolverConfig()

            # Validate the config values after all default-initialization is done
            # to ensure that any inter-dependent parameters are properly checked.
            self.validate()

    _kamino = None
    """
    Class variable storing the imported Kamino module.\n
    The module is imported and cached on the first instantiation of
    the solver to avoid import overhead if the solver is not used.
    """

    def __init__(
        self,
        model: Model,
        config: Config | None = None,
    ):
        """
        Constructs a Kamino solver for the given model and optional configurations.

        Args:
            model:
                The Newton model for which to create the Kamino solver instance.
            config:
                Explicit user-provided configurations for the Kamino solver.\n
                If `None`, configurations will be parsed from the Newton model's
                custom attributes using :meth:`SolverKamino.Config.from_model`,
                e.g. to be loaded from USD assets. If that also fails, then
                default configurations will be used.
        """
        # Initialize the base solver
        super().__init__(model=model)

        # Import all Kamino dependencies and cache them
        # as class variables if not already done
        self._import_kamino()

        # Validate that the model does not contain unsupported components
        self._validate_model_compatibility(model)

        # Cache configurations; either from the user-provided config or from the model's custom attributes
        # NOTE: `Config.from_model` will default-initialize if no relevant custom attributes were
        # found on the model, so `self._config` will always be fully initialized after this step.
        if config is None:
            config = self.Config.from_model(model)
        self._config = config

        # Create a Kamino model from the Newton model
        self._model_kamino = self._kamino.ModelKamino.from_newton(model)

        # Create a collision detector if enabled in the config, otherwise
        # set to `None` to disable internal collision detection in Kamino
        self._collision_detector_kamino = None
        if self._config.use_collision_detector:
            self._collision_detector_kamino = self._kamino.CollisionDetector(
                model=self._model_kamino,
                config=self._config.collision_detector,
            )

        # Capture a reference to the contacts container
        self._contacts_kamino = None
        if self._collision_detector_kamino is not None:
            self._contacts_kamino = self._collision_detector_kamino.contacts
        else:
            # If collision detector is disabled allocate contacts manually
            # TODO: We need to fix this logic to properly handle the case where the collision
            # detector is disabled but contacts are still provided by Newton's collision pipeline.
            if self.model.rigid_contact_max == 0:
                world_max_contacts = self._model_kamino.geoms.world_minimum_contacts
            else:
                world_max_contacts = [model.rigid_contact_max // self.model.world_count] * self.model.world_count
            self._contacts_kamino = self._kamino.ContactsKamino(capacity=world_max_contacts, device=self.model.device)

        # Initialize the internal Kamino solver
        self._solver_kamino = self._kamino.SolverKaminoImpl(
            model=self._model_kamino,
            contacts=self._contacts_kamino,
            config=self._config,
        )

    def reset(
        self,
        state_out: State,
        world_mask: wp.array | None = None,
        actuator_q: wp.array | None = None,
        actuator_u: wp.array | None = None,
        joint_q: wp.array | None = None,
        joint_u: wp.array | None = None,
        base_q: wp.array | None = None,
        base_u: wp.array | None = None,
    ):
        """
        Resets the simulation state given a combination of desired base body
        and joint states, as well as an optional per-world mask array indicating
        which worlds should be reset. The reset state is written to `state_out`.

        For resets given absolute quantities like base body poses, the
        `state_out` must initially contain the current state of the simulation.

        Args:
            state_out: The output state container to which the reset state data is written.
            world_mask: Optional array of per-world masks indicating which worlds should be reset.\n
                Shape of `(num_worlds,)` and type :class:`wp.int8 | wp.bool`
            actuator_q: Optional array of target actuated joint coordinates.\n
                Shape of `(num_actuated_joint_coords,)` and type :class:`wp.float32`
            actuator_u: Optional array of target actuated joint DoF velocities.\n
                Shape of `(num_actuated_joint_dofs,)` and type :class:`wp.float32`
            joint_q: Optional array of target joint coordinates.\n
                Shape of `(num_joint_coords,)` and type :class:`wp.float32`
            joint_u: Optional array of target joint DoF velocities.\n
                Shape of `(num_joint_dofs,)` and type :class:`wp.float32`
            base_q: Optional array of target base body poses.\n
                Shape of `(num_worlds,)` and type :class:`wp.transformf`
            base_u: Optional array of target base body twists.\n
                Shape of `(num_worlds,)` and type :class:`wp.spatial_vectorf`
        """
        # Convert base pose from body-origin to COM frame
        if base_q is not None:
            base_q_com = wp.zeros_like(base_q)
            self._kamino.convert_base_origin_to_com(
                base_body_index=self._model_kamino.info.base_body_index,
                body_com=self._model_kamino.bodies.i_r_com_i,
                base_q=base_q,
                base_q_com=base_q_com,
            )
            base_q = base_q_com

        # TODO: fix brittle in-place update of arrays after conversion
        # Create a zer-copy view of the input state_out as a StateKamino
        # to interface with the Kamino solver's reset operation
        state_out_kamino = self._kamino.StateKamino.from_newton(self._model_kamino.size, self.model, state_out)

        # Execute the reset operation of the Kamino solver,
        # to write the reset state to `state_out_kamino`
        self._solver_kamino.reset(
            state_out=state_out_kamino,
            world_mask=world_mask,
            actuator_q=actuator_q,
            actuator_u=actuator_u,
            joint_q=joint_q,
            joint_u=joint_u,
            base_q=base_q,
            base_u=base_u,
        )

        # Convert com-frame poses from Kamino reset to body-origin frame
        self._kamino.convert_body_com_to_origin(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q_com=state_out_kamino.q_i,
            body_q=state_out_kamino.q_i,
            world_mask=world_mask,
            body_wid=self._model_kamino.bodies.wid,
        )

    @override
    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        """
        Simulate the model for a given time step using the given control input.

        When ``contacts`` is not ``None`` (i.e. produced by :meth:`~newton.Model.collide`),
        those contacts are converted to Kamino's internal format and used directly,
        bypassing Kamino's own collision detector.  When ``contacts`` is ``None``,
        Kamino's internal collision pipeline runs as a fallback.

        Args:
            state_in: The input state.
            state_out: The output state.
            control: The control input.
                Defaults to `None` which means the control values from the
                :class:`Model` are used.
            contacts: The contact information from Newton's collision
                pipeline, or ``None`` to use Kamino's internal collision detector.
            dt: The time step (typically in seconds).
        """
        # Interface the input state containers to Kamino's equivalents
        # NOTE: These should produce zero-copy views/references
        # to the arrays of the source Newton containers.
        state_in_kamino = self._kamino.StateKamino.from_newton(self._model_kamino.size, self.model, state_in)
        state_out_kamino = self._kamino.StateKamino.from_newton(self._model_kamino.size, self.model, state_out)

        # Handle the control input, defaulting to the model's
        # internal control arrays if None is provided.
        if control is None:
            control = self.model.control(clone_variables=False)
        control_kamino = self._kamino.ControlKamino.from_newton(control)

        # If contacts are provided, use them directly, bypassing Kamino's collision detector
        if contacts is not None:
            self._kamino.convert_contacts_newton_to_kamino(self.model, state_in, contacts, self._contacts_kamino)
            _detector = None
        # Otherwise, use Kamino's internal collision detector to generate contacts
        else:
            _detector = self._collision_detector_kamino

        # Convert Newton body-frame poses to Kamino CoM-frame poses using
        # Kamino's corrected body-com offsets (can differ from Newton model data).
        self._kamino.convert_body_origin_to_com(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q=state_in_kamino.q_i,
            body_q_com=state_in_kamino.q_i,
        )

        # Step the physics solver
        self._solver_kamino.step(
            state_in=state_in_kamino,
            state_out=state_out_kamino,
            control=control_kamino,
            contacts=self._contacts_kamino,
            detector=_detector,
            dt=dt,
        )

        # Convert back from Kamino CoM-frame to Newton body-frame poses using
        # the same corrected body-com offsets as the forward conversion.
        self._kamino.convert_body_com_to_origin(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q_com=state_in_kamino.q_i,
            body_q=state_in_kamino.q_i,
        )
        self._kamino.convert_body_com_to_origin(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q_com=state_out_kamino.q_i,
            body_q=state_out_kamino.q_i,
        )

    @override
    def notify_model_changed(self, flags: int) -> None:
        """Propagate Newton model property changes to Kamino's internal ModelKamino.

        Args:
            flags: Bitmask of :class:`SolverNotifyFlags` indicating which properties changed.
        """
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self._update_gravity()

        if flags & SolverNotifyFlags.BODY_PROPERTIES:
            pass  # TODO: convert to CoM-frame if body_q_i_0 is changed at runtime?

        if flags & SolverNotifyFlags.BODY_INERTIAL_PROPERTIES:
            # Kamino's RigidBodiesModel references Newton's arrays directly
            # (m_i, inv_m_i, i_I_i, inv_i_I_i, i_r_com_i), so no copy needed.
            pass

        if flags & SolverNotifyFlags.SHAPE_PROPERTIES:
            pass  # TODO: ???

        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self._update_joint_transforms()

        if flags & SolverNotifyFlags.JOINT_DOF_PROPERTIES:
            # Joint limits (q_j_min, q_j_max, dq_j_max, tau_j_max) are direct
            # references to Newton's arrays, so no copy needed.
            pass

        if flags & SolverNotifyFlags.ACTUATOR_PROPERTIES:
            pass  # TODO: ???

        if flags & SolverNotifyFlags.CONSTRAINT_PROPERTIES:
            pass  # TODO: ???

        unsupported = flags & ~(
            SolverNotifyFlags.MODEL_PROPERTIES
            | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            | SolverNotifyFlags.JOINT_PROPERTIES
            | SolverNotifyFlags.JOINT_DOF_PROPERTIES
        )
        if unsupported:
            self._kamino.msg.warning(
                "SolverKamino.notify_model_changed: flags 0x%x not yet supported",
                unsupported,
            )

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """
        Converts Kamino contacts to Newton's Contacts format.

        Args:
            contacts: The Newton Contacts object to populate.
            state: Simulation state providing ``body_q`` for converting
                world-space contact positions to body-local frame.
        """
        # Ensure the containers are not None and of the correct shape
        if contacts is None:
            raise ValueError("contacts cannot be None when calling SolverKamino.update_contacts")
        elif not isinstance(contacts, Contacts):
            raise TypeError(f"contacts must be of type Contacts, got {type(contacts)}")
        if state is None:
            raise ValueError("state cannot be None when calling SolverKamino.update_contacts")
        elif not isinstance(state, State):
            raise TypeError(f"state must be of type State, got {type(state)}")

        # Skip the conversion if contacts have not been allocated
        if self._contacts_kamino is None or self._contacts_kamino._data.model_max_contacts_host == 0:
            return

        # Ensure the output contacts containers has sufficient size to hold the contact data from Kamino
        if self._contacts_kamino._data.model_max_contacts_host > contacts.rigid_contact_max:
            raise ValueError(
                f"Contacts container has insufficient capacity for Kamino contacts: "
                f"model_max_contacts={self._contacts_kamino._data.model_max_contacts_host} > "
                f"contacts.rigid_contact_max={contacts.rigid_contact_max}"
            )

        # If all checks pass, proceed to convert contacts from Kamino to Newton format
        self._kamino.convert_contacts_kamino_to_newton(self.model, state, self._contacts_kamino, contacts)

    @override
    @staticmethod
    def register_custom_attributes(builder: ModelBuilder) -> None:
        """
        Register custom attributes for SolverKamino.

        Args:
            builder: The model builder to register the custom attributes to.
        """
        # Register State attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_f_total",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.BODY,
                dtype=wp.spatial_vectorf,
                default=wp.spatial_vectorf(0.0),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_q_prev",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.JOINT_COORD,
                dtype=wp.float32,
                default=0.0,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_lambdas",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.float32,
                default=0.0,
            )
        )

        # Register KaminoSceneAPI attributes so the USD importer will store them on the model
        SolverKamino.Config.register_custom_attributes(builder)

    ###
    # Internals
    ###

    @classmethod
    def _import_kamino(cls):
        """Import the Kamino dependencies and cache them as class variables."""
        if cls._kamino is None:
            try:
                with warnings.catch_warnings():
                    # Set a filter to make all ImportWarnings "always" appear
                    # This is useful to debug import errors on Windows, for example
                    warnings.simplefilter("always", category=ImportWarning)

                    from . import _src as kamino  # noqa: PLC0415

                    cls._kamino = kamino

            except ImportError as e:
                raise ImportError("Kamino backend not found.") from e

    @staticmethod
    def _validate_model_compatibility(model: Model):
        """
        Validates that the model does not contain components unsupported by SolverKamino:
        - particles
        - springs
        - triangles, edges, tetrahedra
        - muscles
        - equality constraints
        - distance, cable, or gimbal joints

        Args:
            model: The Newton model to validate.

        Raises:
            ValueError: If the model contains unsupported components.
        """

        unsupported_features = []
        if model.particle_count > 0:
            unsupported_features.append(f"particles (found {model.particle_count})")
        if model.spring_count > 0:
            unsupported_features.append(f"springs (found {model.spring_count})")
        if model.tri_count > 0:
            unsupported_features.append(f"triangle elements (found {model.tri_count})")
        if model.edge_count > 0:
            unsupported_features.append(f"edge elements (found {model.edge_count})")
        if model.tet_count > 0:
            unsupported_features.append(f"tetrahedral elements (found {model.tet_count})")
        if model.muscle_count > 0:
            unsupported_features.append(f"muscles (found {model.muscle_count})")
        if model.equality_constraint_count > 0:
            unsupported_features.append(f"equality constraints (found {model.equality_constraint_count})")

        # Check for unsupported joint types
        if model.joint_count > 0:
            joint_type_np = model.joint_type.numpy()
            joint_dof_dim_np = model.joint_dof_dim.numpy()
            joint_q_start_np = model.joint_q_start.numpy()
            joint_qd_start_np = model.joint_qd_start.numpy()

            unsupported_joint_types = {}

            for j in range(model.joint_count):
                joint_type = int(joint_type_np[j])
                dof_dim = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
                q_count = int(joint_q_start_np[j + 1] - joint_q_start_np[j])
                qd_count = int(joint_qd_start_np[j + 1] - joint_qd_start_np[j])

                # Check for explicitly unsupported joint types
                if joint_type == JointType.DISTANCE:
                    unsupported_joint_types["DISTANCE"] = unsupported_joint_types.get("DISTANCE", 0) + 1
                elif joint_type == JointType.CABLE:
                    unsupported_joint_types["CABLE"] = unsupported_joint_types.get("CABLE", 0) + 1
                # Check for GIMBAL configuration (3 coords, 3 DoFs, 0 linear/3 angular)
                elif joint_type == JointType.D6 and q_count == 3 and qd_count == 3 and dof_dim == (0, 3):
                    unsupported_joint_types["D6 (GIMBAL)"] = unsupported_joint_types.get("D6 (GIMBAL)", 0) + 1

            if len(unsupported_joint_types) > 0:
                joint_desc = [f"{name} ({count} instances)" for name, count in unsupported_joint_types.items()]
                unsupported_features.append("joint types: " + ", ".join(joint_desc))

        # If any unsupported features were found, raise an error
        if len(unsupported_features) > 0:
            error_msg = "SolverKamino cannot simulate this model due to unsupported features:"
            for feature in unsupported_features:
                error_msg += "\n  - " + feature
            raise ValueError(error_msg)

    def _update_gravity(self):
        """
        Updates Kamino's :class:`GravityModel` from Newton's model.gravity.

        Called when :data:`SolverNotifyFlags.MODEL_PROPERTIES` is raised,
        indicating that ``model.gravity`` may have changed at runtime.
        """
        self._kamino.convert_model_gravity(self.model, self._model_kamino.gravity)

    def _update_joint_transforms(self):
        """
        Re-derive Kamino joint anchors and axes from Newton's joint_X_p / joint_X_c.

        Called when :data:`SolverNotifyFlags.JOINT_PROPERTIES` is raised,
        indicating that ``model.joint_X_p`` or ``model.joint_X_c`` may have
        changed at runtime (e.g. animated root transforms).
        """
        self._kamino.convert_model_joint_transforms(self.model, self._model_kamino.joints)
