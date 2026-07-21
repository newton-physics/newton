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

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import (
    Contacts,
    Control,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
)
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase

if TYPE_CHECKING:
    from .config import (
        CollisionDetectorConfig,
        ConfigBase,
        ConstrainedDynamicsConfig,
        ConstraintStabilizationConfig,
        ForwardKinematicsSolverConfig,
        MaterialManagerConfig,
        PADMMSolverConfig,
    )

###
# Module interface
###

__all__ = ["SolverKamino"]


###
# Interfaces
###


class SolverKamino(SolverBase, CouplingInterface):
    """
    A physics solver for simulating constrained multi-body systems containing kinematic loops,
    under-/overactuation, joint-limits, hard frictional contacts and restitutive impacts.

    This solver uses the Proximal-ADMM algorithm to solve the forward dynamics formulated
    as a Nonlinear Complementarity Problem (NCP) over the set of bilateral kinematic joint
    constraints and unilateral constraints that include joint-limits and contacts.

    This solver is currently in Beta.

    .. experimental::
        SolverKamino's public API and internal implementation may change without
        prior notice, including simulation feature support, performance, and bug fixes.

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
            solver = newton.solvers.SolverKamino(model, config=config)

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

        materials: MaterialManagerConfig | None = None
        """
        Configurations for the material manager and material property mixing.
        See :class:`MaterialManagerConfig` for more details.
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
            config.MaterialManagerConfig.register_custom_attributes(builder)

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
                "materials": config.MaterialManagerConfig,
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
            self.materials.validate()

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
            if self.materials is None:
                self.materials = config.MaterialManagerConfig()

            # Validate the config values after all default-initialization is done
            # to ensure that any inter-dependent parameters are properly checked.
            self.validate()

    _kamino = None
    """
    Class variable storing the imported Kamino module.\n
    The module is imported and cached on the first instantiation of
    the solver to avoid import overhead if the solver is not used.
    """

    @dataclass
    class ResetConfig:
        """
        Configuration for a call to the reset() operation, specifying the behaviour (common or separate)
        for body poses, body velocities as well as floating base pose and velocity.

        Example
        -------

            .. code-block:: python

                # Reset all worlds to the initial state
                reset_config = newton.solvers.SolverKamino.ResetConfig.to_default()
                solver.reset(state=state, config=reset_config)

                # Preserve the current body/joint state, while resetting time, forces/torques and solver internals
                reset_config = newton.solvers.SolverKamino.ResetConfig.preserve()
                solver.reset(state=state, config=reset_config)

                # Set a custom pose from joint state with FK
                wp.copy(state.joint_q, custom_joint_coords)
                wp.copy(state.joint_qd, custom_joint_velocities)
                reset_config = newton.solvers.SolverKamino.ResetConfig.from_joints()
                solver.reset(state=state, config=reset_config)

                # Advanced reset with custom configuration
                # E.g. here, set custom actuator coords and base pose, and reset velocities to default (=zero)
                reset_config = newton.solvers.SolverKamino.ResetConfig(
                    body_poses=newton.solvers.SolverKamino.ResetConfig.FromActuatorQ(new_actuator_coords),
                    body_velocities=newton.solvers.SolverKamino.ResetConfig.ToDefault(),
                    base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(new_base_pose),
                    base_velocity=newton.solvers.SolverKamino.ResetConfig.ToDefault(),
                )
                solver.reset(state=state, config=reset_config)
        """

        @dataclass(frozen=True)
        class ToDefault:
            """Reset option, to reset to default values (e.g., initial pose and zero velocity)."""

        @dataclass(frozen=True)
        class Preserve:
            """Reset option, to preserve current values, assuming without check that they are consistent."""

        @dataclass(frozen=True)
        class FromJointQ:
            """
            Reset option, to set body poses from actuator coordinates and/or base joint coordinates.
            Extracts relevant data from joint coordinates, and applies position-level Forward Kinematics
            and/or a global transformation at the base.
            Note: angles outside the [-2pi, 2pi] range around initial coordinates will be remapped automatically.
            """

            joint_q: wp.array[wp.float32] | None = None
            """Optional joint coordinates array. If not provided, coordinates in the state container are used."""

        @dataclass(frozen=True)
        class FromJointU:
            """
            Reset option, to set body velocities from actuator velocities and/or base joint velocity.
            Extracts relevant data from joint velocities, and applies velocity-level Forward Kinematics
            and/or a global composition with the base velocity.
            """

            joint_u: wp.array[wp.float32] | None = None
            """Optional joint velocities array. If not provided, velocities in the state container are used."""

        @dataclass(frozen=True)
        class FromActuatorQ:
            """
            Reset option, to set body poses from actuator coordinates, using position-level Forward Kinematics.
            Note: angles outside the [-2pi, 2pi] range around initial coordinates will be remapped automatically.
            """

            actuator_q: wp.array[wp.float32]
            """Actuator coordinates array."""

        @dataclass(frozen=True)
        class FromActuatorU:
            """
            Reset option, to set body velocities from actuator velocities, using velocity-level Forward Kinematics.
            """

            actuator_u: wp.array[wp.float32]
            """Actuator velocities array."""

        @dataclass(frozen=True)
        class FromBaseQ:
            """
            Reset option, to set a new pose for the base body, and transform all bodies accordingly.
            If a base joint is set, the prescribed pose is interpreted in the frame of the base joint;
            else it is directly interpreted as the new pose of the base body.
            """

            base_q: wp.array[wp.transformf]
            """Per-world base body pose array."""

        @dataclass(frozen=True)
        class FromBaseU:
            """
            Reset option, to set a new velocity for the base body, and compose with body velocities accordingly.
            If a base joint is set, the prescribed velocity is interpreted in the frame of the base joint;
            else it is directly interpreted as the new velocity of the base body.
            """

            base_u: wp.array[wp.spatial_vectorf]
            """Per-world base body velocity array."""

        body_poses: ToDefault | Preserve | FromJointQ | FromActuatorQ = ToDefault()
        """
        Reset option for body poses:

        - ToDefault: reset poses to their initial values.
        - Preserve: preserve poses in the state container, assuming they are consistent.
        - FromJointQ: extract actuator coordinates from joint coordinates, and compute consistent
          body poses with a position-level forward kinematics solve.
        - FromActuatorQ: compute consistent body poses for the prescribed actuator coordinates with
          a position-level forward kinematics solve.
        """

        body_velocities: ToDefault | Preserve | FromJointU | FromActuatorU = ToDefault()
        """
        Reset option for body velocities:

        - ToDefault: reset velocities to zero.
        - Preserve: if body poses are preserved, preserve velocities in the state container, assuming
          they are consistent. Otherwise, behaves like FromJointU, transferring current joint velocities
          in the state container, to the extent possible, to the new body poses.
        - FromJointU: extract actuator velocities from joint velocities, and compute consistent body
          velocities with a velocity-level forward kinematics solve.
        - FromActuatorU: compute consistent body velocities for the prescribed actuator velocities with
          a velocity-level forward kinematics solve.
        """

        base_pose: ToDefault | Preserve | FromJointQ | FromBaseQ = ToDefault()
        """
        Reset option for the floating base pose:

        - ToDefault: reset the base pose to its initial value.
        - Preserve: preserve the current base pose, as read from current joint coordinates (if a base joint
          was set) or body poses (otherwise).
        - FromJointQ: read the base pose from joint coordinates, assuming a base joint was set. Behaves
          like ToDefault otherwise (as a fallback).
        - FromBaseQ: use the provided base pose.

        Body poses and velocities are transformed (if needed) to match the prescribed base pose, while
        preserving relative poses and velocities.
        All options are ignored for worlds for which no base body is set.
        """

        base_velocity: ToDefault | Preserve | FromJointU | FromBaseU = ToDefault()
        """
        Reset option for the floating base velocity:

        - ToDefault: reset the base velocity to zero.
        - Preserve: preserve the current base velocity, as read from current joint velocities (if a base joint
          was set) or body velocities (otherwise), up to transformation due to the new base pose if applicable.
        - FromJointU: read the base velocity from joint velocities, assuming a base joint was set. Behaves
          like ToDefault otherwise (as a fallback).
        - FromBaseU: use the provided base velocity.

        Body velocities are updated to match the prescribed base velocity, while preserving relative velocities.
        All options are ignored for worlds for which no base body is set.
        """

        @classmethod
        def to_default(cls) -> SolverKamino.ResetConfig:
            """Instantiates a reset config for resetting all state components to default values."""
            return cls(
                body_poses=SolverKamino.ResetConfig.ToDefault(),
                body_velocities=SolverKamino.ResetConfig.ToDefault(),
                base_pose=SolverKamino.ResetConfig.ToDefault(),
                base_velocity=SolverKamino.ResetConfig.ToDefault(),
            )

        @classmethod
        def preserve(cls) -> SolverKamino.ResetConfig:
            """Instantiates a reset config for preserving all state components."""
            return cls(
                body_poses=SolverKamino.ResetConfig.Preserve(),
                body_velocities=SolverKamino.ResetConfig.Preserve(),
                base_pose=SolverKamino.ResetConfig.Preserve(),
                base_velocity=SolverKamino.ResetConfig.Preserve(),
            )

        @classmethod
        def from_joints(cls) -> SolverKamino.ResetConfig:
            """
            Instantiates a reset config for running FK at the position and velocity level,
            to set new poses and velocities from current per-joint values in the state container.
            """
            return cls(
                body_poses=SolverKamino.ResetConfig.FromJointQ(),
                body_velocities=SolverKamino.ResetConfig.FromJointU(),
                base_pose=SolverKamino.ResetConfig.FromJointQ(),
                base_velocity=SolverKamino.ResetConfig.FromJointU(),
            )

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

        # Store for which joints the limits are finite. This is used to validate that finiteness of limits is not changed at runtime.
        q_min = self._model_kamino.joints.q_j_min.numpy()
        q_max = self._model_kamino.joints.q_j_max.numpy()
        built_limit_finite_np = (q_min > self._kamino.JOINT_QMIN) | (q_max < self._kamino.JOINT_QMAX)
        self._built_limit_finite = wp.array(
            built_limit_finite_np.astype(np.int32),
            dtype=wp.int32,
            device=model.device,
        )

        # Scratch array for notify validation
        self._notify_violations = wp.empty(4, dtype=wp.int32, device=model.device)

        # Cache one representative shape per material.
        self._material_first_shape = self._kamino.compute_material_first_shape(
            self._model_kamino.geoms.material,
            self._model_kamino.materials.num_materials,
        )
        # Scratch scalar for material update validation
        self._material_update_conflict = wp.empty(1, dtype=wp.int32, device=model.device)

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
            self._contacts_kamino = self._kamino.ContactsKamino(
                # TODO: model=self._model_kamino,
                capacity=world_max_contacts,
                device=self.model.device,
                remappable=True,
            )

        # Declare an internal reference cache to be able to detect if
        # a Kamino-internal collision detector was used at runtime.
        # NOTE: This is used to determine whether to clear the output
        # contacts and populate them with only active contacts or fill
        # in solver-specific contact attributes for existing contacts.
        # TODO: Do we need this additional indirection or is there a better way to do this?
        self._detector = None

        # Initialize the internal Kamino solver
        self._solver_kamino = self._kamino.SolverKaminoImpl(
            model=self._model_kamino,
            contacts=self._contacts_kamino,
            config=self._config,
        )

        # Initialize the internal Kamino control wrapper
        self._control_kamino = self._kamino.ControlKamino()
        self._control_kamino.finalize(self._model_kamino)

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
        *,
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
            flags: Optional :class:`~newton.StateFlags` or ``int`` bitmask controlling
                which state attributes need to be reset.  If ``None``, all
                state attributes are reset.
                Note: currently, this is implementing simply by caching attributes that
                should not be reset, and restoring them after the Kamino-internal reset.
                For complex/partial resets, it is recommended to use config instead.
            config: Optional reset configuration, controlling the reset behavior
                for body poses/velocities as well as floating base pose/velocity.
                If not provided, all components are reset to default (initial) values.
            success_mask: Optional mask, filled with a success boolean per world if provided
                (True if reset successfully, False if not reset due to world_mask, or if reset
                was unsuccessful, e.g. due to an unconverged FK solve).
        """
        if state is None:
            raise ValueError("'state' argument is required.")

        # Process None arguments
        state_flags = int(StateFlags.ALL if flags is None else flags)
        config = SolverKamino.ResetConfig.to_default() if config is None else config

        # Convert/alias the input state as a StateKamino object
        state_kamino = self._kamino.StateKamino.from_newton(
            self._model_kamino.size, self.model, state, convert_to_com_frame=False
        )

        # Convert body poses from origin to CoM if needed
        has_callbacks = self._solver_kamino._pre_reset_cb is not None or self._solver_kamino._post_reset_cb is not None
        need_CoM_conversion = (
            not isinstance(config.body_poses, SolverKamino.ResetConfig.Preserve)
            or not isinstance(config.base_pose, SolverKamino.ResetConfig.Preserve)
            or has_callbacks
        )
        if need_CoM_conversion:
            self._kamino.convert_body_origin_to_com(
                body_com=self._model_kamino.bodies.i_r_com_i,
                body_q_com=state_kamino.q_i,
                body_q=state_kamino.q_i,
                world_mask=world_mask if not has_callbacks else None,
                body_wid=self._model_kamino.bodies.wid,
            )
            # Note: we convert all worlds if callbacks are set, so they see the full state correctly

        # Convert base pose from origin to CoM if needed
        if isinstance(config.base_pose, SolverKamino.ResetConfig.FromBaseQ):
            base_q_com = wp.zeros_like(config.base_pose.base_q)
            self._kamino.convert_base_origin_to_com(
                base_joint_index=self._model_kamino.info.base_joint_index,
                base_body_index=self._model_kamino.info.base_body_index,
                body_com=self._model_kamino.bodies.i_r_com_i,
                base_q=config.base_pose.base_q,
                base_q_com=base_q_com,
            )
            config_cache = config.base_pose
            config.base_pose = SolverKamino.ResetConfig.FromBaseQ(base_q_com)

        # Cache fields excluded from the reset op, to restore them afterwards
        restore_after_reset: list[tuple[wp.array, wp.array]] = []

        def _preserve_if_unset(array: wp.array[Any] | None, flag: int) -> None:
            if array is not None and not (state_flags & flag):
                restore_after_reset.append((array, wp.clone(array, device=array.device)))

        _preserve_if_unset(state_kamino.q_j, StateFlags.JOINT_Q)
        _preserve_if_unset(state_kamino.q_j_p, StateFlags.JOINT_Q)
        _preserve_if_unset(state_kamino.dq_j, StateFlags.JOINT_QD)
        _preserve_if_unset(state_kamino.q_i, StateFlags.BODY_Q)
        _preserve_if_unset(state_kamino.u_i, StateFlags.BODY_QD)

        # Execute the reset operation of the Kamino solver,
        # to write the reset state to `state_kamino`.
        self._solver_kamino.reset(
            state=state_kamino,
            world_mask=world_mask,
            config=config,
            success_mask=success_mask,
        )

        # Restore fields excluded from the reset op
        for array, snapshot in restore_after_reset:
            wp.copy(array, snapshot)

        # Convert back body poses from COM-frame (Kamino) to body-origin frame (Newton)
        if need_CoM_conversion:
            self._kamino.convert_body_com_to_origin(
                body_com=self._model_kamino.bodies.i_r_com_i,
                body_q_com=state_kamino.q_i,
                body_q=state_kamino.q_i,
                world_mask=world_mask if not has_callbacks else None,
                body_wid=self._model_kamino.bodies.wid,
            )

        # Revert changes to config
        if isinstance(config.base_pose, SolverKamino.ResetConfig.FromBaseQ):
            config.base_pose = config_cache

    @override
    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        """
        Simulate the model for a given time step using the given control input.

        When ``contacts`` is not ``None`` (i.e. populated by :meth:`~newton.CollisionPipeline.collide`),
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
        self._control_kamino.from_newton(control, self._model_kamino)

        # If contacts are provided, use them directly, bypassing Kamino's collision detector
        if contacts is not None:
            self._detector = None
            self._kamino.convert_contacts_newton_to_kamino(
                model=self.model,
                state=state_in,
                contacts_in=contacts,
                contacts_out=self._contacts_kamino,
                convert_forces=False,
                friction_mix_mode=self._config.materials.friction_mix_mode,
                restitution_mix_mode=self._config.materials.restitution_mix_mode,
            )
        # Otherwise, use Kamino's internal collision detector to generate contacts
        else:
            self._detector = self._collision_detector_kamino

        # Convert Newton body-frame poses to Kamino CoM-frame poses
        self._kamino.convert_body_origin_to_com(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q=state_in_kamino.q_i,
            body_q_com=state_in_kamino.q_i,
        )

        # Step the physics solver
        self._solver_kamino.step(
            state_in=state_in_kamino,
            state_out=state_out_kamino,
            control=self._control_kamino,
            contacts=self._contacts_kamino,
            detector=self._detector,
            dt=dt,
        )

        # Convert back from Kamino CoM-frame to Newton body-frame poses
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
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Propagate Newton model property changes to Kamino's internal ModelKamino.

        Args:
            flags: Bitmask of :class:`~newton.ModelFlags` or custom ``int`` bits indicating which properties changed.
        """
        self._validate_structural_invariants(flags)

        if flags & (ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.ACTUATOR_PROPERTIES):
            # The documentation is unclear about which flag should trigger this update, so we update on both flags.
            self._update_actuation_types()

        if flags & ModelFlags.MODEL_PROPERTIES:
            self._update_gravity()

        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            # q_i_0 is derived from both model.body_q and model.body_com.
            self._update_body_initial_pose()

        if flags & (ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.JOINT_PROPERTIES):
            # Joint transforms are derived from body_com and joint_X_p / joint_X_c.
            self._update_joint_transforms()

        if flags & (ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.SHAPE_PROPERTIES):
            # Geom offsets are derived from body_com and shape_transform.
            self._update_geom_offsets()

        if flags & ModelFlags.SHAPE_PROPERTIES:
            self._update_materials()

        if flags & (ModelFlags.CONSTRAINT_PROPERTIES | ModelFlags.TENDON_PROPERTIES):
            # Kamino does not support equality/mimic constraints or tendons, so we ignore these flags.
            # When using a coupled solver environment, these flags are meant for one of the other solvers.
            # No warning is emitted for compatibility with such an environment.
            pass

        handled = (
            ModelFlags.MODEL_PROPERTIES
            | ModelFlags.BODY_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.ACTUATOR_PROPERTIES
            | ModelFlags.CONSTRAINT_PROPERTIES
            | ModelFlags.TENDON_PROPERTIES
        )
        unsupported = int(flags) & ~int(handled)
        if unsupported:
            self._kamino.msg.warning(
                "SolverKamino.notify_model_changed: flags 0x%x not yet supported",
                unsupported,
            )

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """
        Converts Kamino contacts to Newton's Contacts format.

        Note: produces undefined behavior if a different Newton Contacts object was
        passed to step().

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
        if self._contacts_kamino is None or self._contacts_kamino.model_max_contacts_host == 0:
            return

        # Ensure the output contacts containers has sufficient size to hold the contact data from Kamino
        if self._contacts_kamino.model_max_contacts_host > contacts.rigid_contact_max:
            raise RuntimeError(
                f"Contacts container has insufficient capacity for Kamino contacts: "
                f"model_max_contacts={self._contacts_kamino.model_max_contacts_host} > "
                f"contacts.rigid_contact_max={contacts.rigid_contact_max}"
            )

        # If all checks pass, proceed to convert contacts from Kamino to Newton format
        self._kamino.convert_contacts_kamino_to_newton(
            model=self.model,
            state=state,
            contacts_in=self._contacts_kamino,
            contacts_out=contacts,
            clear_output=self._detector is not None,
            convert_forces=True,
        )

    @override
    @staticmethod
    def register_custom_attributes(
        builder: ModelBuilder,
        *,
        fk_actuation_flags: dict[int, int] | None = None,
    ) -> None:
        """
        Register custom attributes for SolverKamino.

        Args:
            builder: The model builder to register the custom attributes to.
            fk_actuation_flags: Optional dictionary of {joint_index: fk_actuation_flag} integer flags,
                overwriting what joints should be considered actuated (flag = 1) or passive (flag = 0)
                by the Forward Kinematics solver during reset() operations.
                Joints not listed or with a flag of -1 use the joint actuation type from the model
                (treating all actuator types equally, as only passive vs actuated matters in FK).
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

        # Register FK custom actuation types
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="fk_actuation_flag",
                assignment=Model.AttributeAssignment.MODEL,
                frequency=Model.AttributeFrequency.JOINT,
                dtype=wp.int32,
                default=-1,
                values=fk_actuation_flags,
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

    def _validate_structural_invariants(self, flags: ModelFlags | int) -> None:
        """Raise if a runtime edit changes a structural decision frozen at build.

        Kamino freezes joint constraint counts, the actuated/passive partition,
        and joint-limit slot capacity when constructing its model. The underlying
        Newton values may be aliased, but the derived layout cannot change.

        Raises:
            RuntimeError: If the solver must be recreated to apply the edit.
        """
        check_dof = bool(flags & ModelFlags.JOINT_DOF_PROPERTIES)
        check_actuation = bool(flags & (ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.ACTUATOR_PROPERTIES))
        if not check_dof and not check_actuation:
            return

        sentinel = self._kamino.validate_model_joint_updates(
            self.model,
            self._model_kamino.joints,
            self._built_limit_finite,
            self._notify_violations,
            check_dof=check_dof,
            check_actuation=check_actuation,
        )
        dynamic_joint, limit_dof, actuation_joint, invalid_joint = self._notify_violations.numpy()

        if dynamic_joint != sentinel:
            joint = int(dynamic_joint)
            raise RuntimeError(
                f"Changing dynamic constraint topology for joint {joint} "
                f"({self.model.joint_label[joint]!r}) is not supported; recreate SolverKamino to apply the change. "
                "The dynamic constraint topology changes if armature, damping, target stiffness, or target damping are updated to non-zero values, while they were zero when creating the solver. "
                "The opposite is also true: if the values are updated to zero, while they were non-zero when creating the solver, the dynamic constraint topology also changes."
            )

        if limit_dof != sentinel:
            dof = int(limit_dof)
            raise RuntimeError(
                f"Changing the existence of a joint limit for DoF {dof} "
                f"is not supported; recreate SolverKamino to apply the change."
            )

        if actuation_joint != sentinel:
            joint = int(actuation_joint)
            raise RuntimeError(
                f"Changing the actuation partition for joint {joint} "
                f"({self.model.joint_label[joint]!r}) is not supported; recreate SolverKamino to apply the change."
            )

        if invalid_joint != sentinel:
            joint = int(invalid_joint)
            raise ValueError(f"Unsupported joint target mode for joint {joint}")

    def _update_actuation_types(self) -> None:
        """Refresh actuation modes without changing the passive/actuated layout."""
        self._kamino.convert_model_joint_actuation(self.model, self._model_kamino.joints)

    def _update_gravity(self):
        """Update Kamino's :class:`GravityModel` from Newton's ``model.gravity``."""
        self._kamino.convert_model_gravity(self.model, self._model_kamino.gravity)

    def _update_body_initial_pose(self):
        """Recompute Kamino's CoM-frame initial body poses."""
        self._kamino.convert_body_origin_to_com(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q=self.model.body_q,
            body_q_com=self._model_kamino.bodies.q_i_0,
        )

    def _update_geom_offsets(self):
        """Recompute Kamino's CoM-relative geom offsets."""
        self._kamino.convert_geom_offset_origin_to_com(
            body_com=self._model_kamino.bodies.i_r_com_i,
            geom_bid=self._model_kamino.geoms.bid,
            geom_offset=self.model.shape_transform,
            geom_offset_com=self._model_kamino.geoms.offset,
        )

    def _update_joint_transforms(self):
        """Re-derive Kamino joint anchors and axes from Newton's joint transforms."""
        self._kamino.convert_model_joint_transforms(self.model, self._model_kamino.joints)

    def _update_materials(self) -> None:
        """Refresh Kamino contact-material tables using cached representative shapes."""
        self._kamino.convert_model_materials(
            self.model,
            self._model_kamino,
            self._material_first_shape,
            self._material_update_conflict,
        )
