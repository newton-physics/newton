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
Defines configurations for :class:`SolverKamino`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import warp as wp

from ...core.types import override
from ...sim import Model, ModelBuilder

###
# Module interface
###

__all__ = [
    "CollisionDetectorConfig",
    "ConfigBase",
    "ConstrainedDynamicsConfig",
    "ConstraintStabilizationConfig",
    "ForwardKinematicsSolverConfig",
    "PADMMSolverConfig",
]


###
# Types
###


@dataclass
class ConfigBase:
    """
    Defines a base class for configuration containers providing interfaces for
    registering custom attributes and parsing configurations from a Newton model.
    """

    @staticmethod
    def register_custom_attributes(builder: ModelBuilder) -> None:
        """
        Registers custom attributes for config type with the given builder.

        Args:
            builder: The model builder instance with which to register the custom attributes.
        """
        pass

    @staticmethod
    def from_model(model: Model, **kwargs: dict[str, Any]) -> ConfigBase:
        """
        Creates a :class:`ConfigBase` by attempting to parse custom attributes from a :class:`Model` if available.

        Args:
            model: The Newton model from which to parse configurations.
        """
        return ConfigBase(**kwargs)

    def validate(self) -> None:
        """
        Validates the config parameters to ensure they are within acceptable ranges and consistent with each other.

        Raises:
            ValueError: If any parameter is out of range or if there are inconsistencies between parameters.
            TypeError: If any parameter is of an incorrect type.
        """
        pass


@dataclass
class CollisionDetectorConfig(ConfigBase):
    """
    A container to hold configurations for the internal collision detector used for contact generation.
    """

    pipeline: Literal["primitive", "unified"] = "unified"
    """
    The type of collision-detection pipeline to use, either `primitive` or `unified`.\n
    Defaults to `unified`.
    """

    broadphase: Literal["nxn", "sap", "explicit"] = "explicit"
    """
    The broad-phase collision-detection to use (`nxn`, `sap`, or `explicit`).\n
    Defaults to `explicit`.
    """

    bvtype: Literal["aabb", "bs"] = "aabb"
    """
    The type of bounding volume to use in the broad-phase.\n
    Defaults to `aabb`.
    """

    max_contacts: int | None = None
    """
    The maximum number of contacts to generate over the entire model.\n
    Used to compute the total maximum contacts allocated for the model,
    in conjunction with the total number of candidate geom-pairs.\n
    Defaults to `DEFAULT_MODEL_MAX_CONTACTS` (`1000`) if unspecified.
    """

    max_contacts_per_world: int | None = None
    """
    The per-world maximum contacts allocation override.\n
    If specified, it will override the per-world maximum number of contacts
    computed according to the candidate geom-pairs represented in the model.\n
    Defaults to `None`, allowing contact allocations to occur according to the model.
    """

    max_contacts_per_pair: int | None = None
    """
    The maximum number of contacts to generate per candidate geom-pair.\n
    Used to compute the total maximum contacts allocated for the model,
    in conjunction with the total number of candidate geom-pairs.\n
    Defaults to `DEFAULT_GEOM_PAIR_MAX_CONTACTS` (`12`) if unspecified.
    """

    max_triangle_pairs: int | None = None
    """
    The maximum number of triangle-primitive shape pairs to consider in the narrow-phase.\n
    Used only when the model contains triangle meshes or heightfields.\n
    Defaults to `DEFAULT_TRIANGLE_MAX_PAIRS` (`1_000_000`) if unspecified.
    """

    default_gap: float | None = None
    """
    The default detection gap [m] applied as a floor to per-geometry gaps.\n
    Defaults to `DEFAULT_GEOM_PAIR_CONTACT_GAP` (`0.0`) if unspecified.
    """

    @override
    @staticmethod
    def register_custom_attributes(builder: ModelBuilder) -> None:
        """
        Registers custom attributes for the CollisionDetector solver config with the given builder.

        Note: Currently, this class does not have any custom attributes registered,
        as only those supported by the Kamino USD scene API have been included. More
        will be added in the future as latter is being developed.

        Args:
            builder: The model builder instance with which to register the custom attributes.
        """
        pass  # TODO: Add custom attributes for the CD when supported by the Kamino USD scene API

    @override
    @staticmethod
    def from_model(model: Model, **kwargs: dict[str, Any]) -> CollisionDetectorConfig:
        """
        Creates a :class:`CollisionDetectorConfig` by attempting to
        parse custom attributes from a :class:`Model` if available.

        Args:
            model: The Newton model from which to parse configurations.
        """
        cfg = CollisionDetectorConfig(**kwargs)

        # TODO: Implement these

        # Return the fully constructed config with configurations
        # parsed from the model's custom attributes if available,
        # otherwise using defaults or provided kwargs.
        return cfg

    @override
    def validate(self) -> None:
        """
        Validates the current values held by the :class:`CollisionDetectorConfig` instance.
        """
        # Import here to avoid module-level imports and circular dependencies
        from ._src.geometry import BoundingVolumeType, BroadPhaseType, CollisionPipelineType  # noqa: PLC0415
        from ._src.geometry.contacts import (  # noqa: PLC0415
            DEFAULT_GEOM_PAIR_CONTACT_GAP,
            DEFAULT_GEOM_PAIR_MAX_CONTACTS,
            DEFAULT_MODEL_MAX_CONTACTS,
            DEFAULT_TRIANGLE_MAX_PAIRS,
        )

        # Check that the string literals provided correspond to supported enum types, and raise an error if not
        pipelines_supported = [e.name.lower() for e in CollisionPipelineType]
        if self.pipeline not in pipelines_supported:
            raise ValueError(f"Invalid CD pipeline type: {self.pipeline}. Valid options are: {pipelines_supported}")
        broadphases_supported = [e.name.lower() for e in BroadPhaseType]
        if self.broadphase not in broadphases_supported:
            raise ValueError(
                f"Invalid CD broad-phase type: {self.broadphase}. Valid options are: {broadphases_supported}"
            )
        bvtypes_supported = [e.name.lower() for e in BoundingVolumeType]
        if self.bvtype not in bvtypes_supported:
            raise ValueError(f"Invalid CD bounding-volume type: {self.bvtype}. Valid options are: {bvtypes_supported}")

        # Ensure that max_contacts, if specified, is non-negative
        if self.max_contacts is not None and self.max_contacts < 0:
            raise ValueError(f"Invalid max_contacts: {self.max_contacts}. Must be non-negative.")
        if self.max_contacts_per_world is not None and self.max_contacts_per_world < 0:
            raise ValueError(f"Invalid max_contacts_per_world: {self.max_contacts_per_world}. Must be non-negative.")
        if self.max_contacts_per_pair is not None and self.max_contacts_per_pair < 0:
            raise ValueError(f"Invalid max_contacts_per_pair: {self.max_contacts_per_pair}. Must be non-negative.")
        if self.max_triangle_pairs is not None and self.max_triangle_pairs < 0:
            raise ValueError(f"Invalid max_triangle_pairs: {self.max_triangle_pairs}. Must be non-negative.")

        # Check if optional arguments are specified and override with defaults if not
        if self.max_contacts is None:
            self.max_contacts = DEFAULT_MODEL_MAX_CONTACTS
        if self.max_contacts_per_pair is None:
            self.max_contacts_per_pair = DEFAULT_GEOM_PAIR_MAX_CONTACTS
        if self.max_triangle_pairs is None:
            self.max_triangle_pairs = DEFAULT_TRIANGLE_MAX_PAIRS
        if self.default_gap is None:
            self.default_gap = DEFAULT_GEOM_PAIR_CONTACT_GAP

    @override
    def __post_init__(self):
        """Post-initialization to validate configurations."""
        self.validate()


@dataclass
class ConstraintStabilizationConfig(ConfigBase):
    """
    A container to hold configurations for global constraint stabilization parameters.

    These parameters serve as global defaults/overrides, to be used
    in combination with the per-constraint stabilization parameters
    specified in the model, if the latter are provided.
    """

    alpha: float = 0.01
    """
    Global default Baumgarte stabilization parameter for bilateral joint constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    beta: float = 0.01
    """
    Global default Baumgarte stabilization parameter for unilateral joint-limit constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    gamma: float = 0.01
    """
    Global default Baumgarte stabilization parameter for unilateral contact constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    delta: float = 1.0e-6
    """
    Contact penetration margin used for unilateral contact constraints.\n
    Must be non-negative.\n
    Defaults to `1.0e-6`.
    """

    @override
    @staticmethod
    def register_custom_attributes(builder: ModelBuilder) -> None:
        """
        Registers custom attributes for this config with the given builder.

        Note: Currently, not all configurations are registered as custom attributes,
        as only those supported by the Kamino USD scene API have been included. More
        will be added in the future as latter is being developed.

        Args:
            builder: The model builder instance with which to register the custom attributes.
        """
        # Create a default instance of the config to access default values for the attributes
        default_cfg = ConstraintStabilizationConfig()

        # Register KaminoSceneAPI attributes so the USD importer will store them on the model
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="constraints_alpha",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=default_cfg.alpha,
                namespace="kamino",
                usd_attribute_name="newton:kamino:constraints:alpha",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="constraints_beta",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=default_cfg.beta,
                namespace="kamino",
                usd_attribute_name="newton:kamino:constraints:beta",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="constraints_gamma",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=default_cfg.gamma,
                namespace="kamino",
                usd_attribute_name="newton:kamino:constraints:gamma",
            )
        )

    @override
    @staticmethod
    def from_model(model: Model, **kwargs: dict[str, Any]) -> ConstraintStabilizationConfig:
        """
        Creates a :class:`ConstraintStabilizationConfig` by attempting
        to parse custom attributes from a :class:`Model` if available.

        Args:
            model: The Newton model from which to parse configurations.
        """
        cfg = ConstraintStabilizationConfig(**kwargs)

        # Parse solver-specific attributes imported from USD
        kamino_attrs = getattr(model, "kamino", None)
        if kamino_attrs is not None:
            if hasattr(kamino_attrs, "constraints_alpha"):
                cfg.alpha = float(kamino_attrs.constraints_alpha.numpy()[0])
            if hasattr(kamino_attrs, "constraints_beta"):
                cfg.beta = float(kamino_attrs.constraints_beta.numpy()[0])
            if hasattr(kamino_attrs, "constraints_gamma"):
                cfg.gamma = float(kamino_attrs.constraints_gamma.numpy()[0])

        # Return the fully constructed config with configurations
        # parsed from the model's custom attributes if available,
        # otherwise using defaults or provided kwargs.
        return cfg

    @override
    def validate(self) -> None:
        """
        Validates the current values held by the :class:`ConstraintStabilizationConfig` instance.
        """
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError(f"Invalid alpha: {self.alpha}. Must be in range [0, 1.0].")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError(f"Invalid beta: {self.beta}. Must be in range [0, 1.0].")
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError(f"Invalid gamma: {self.gamma}. Must be in range [0, 1.0].")
        if self.delta < 0.0:
            raise ValueError(f"Invalid delta: {self.delta}. Must be non-negative.")

    @override
    def __post_init__(self):
        """Post-initialization to validate configurations."""
        self.validate()


@dataclass
class ConstrainedDynamicsConfig(ConfigBase):
    """
    A container to hold configurations for the construction of the constrained forward dynamics problem.
    """

    preconditioning: bool = True
    """
    Set to `True` to enable preconditioning of the dual problem.\n
    Defaults to `True`.
    """

    linear_solver_type: Literal["LLTB", "CR"] = "LLTB"
    """
    The type of linear solver to use for the dynamics problem.\n
    See :class:`LinearSolverType` for available options.\n
    Defaults to 'LLTB', which will use the :class:`LLTBlockedSolver`.
    """

    linear_solver_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Additional keyword arguments to pass to the linear solver.\n
    Defaults to an empty dictionary.
    """

    @override
    @staticmethod
    def register_custom_attributes(builder: ModelBuilder) -> None:
        """
        Registers custom attributes for the constrained dynamics problem configurations with the given builder.

        Note: Currently, not all configurations are registered as custom attributes,
        as only those supported by the Kamino USD scene API have been included. More
        will be added in the future as latter is being developed.

        Args:
            builder: The model builder instance with which to register the custom attributes.
        """
        # Register KaminoSceneAPI attributes so the USD importer will store them on the model
        # TODO: Rename `name` to this to "dynamics_preconditioning" or similar
        # TODO: Rename `usd_attribute_name` to "newton:kamino:usePreconditioning" or similar
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="constraints_use_preconditioning",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=True,
                namespace="kamino",
                usd_attribute_name="newton:kamino:constraints:usePreconditioning",
            )
        )

    @override
    @staticmethod
    def from_model(model: Model, **kwargs: dict[str, Any]) -> ConstrainedDynamicsConfig:
        """
        Creates a :class:`ConstrainedDynamicsConfig` by attempting to
        parse custom attributes from a :class:`Model` if available.

        Args:
            model: The Newton model from which to parse configurations.
        """
        cfg = ConstrainedDynamicsConfig(**kwargs)

        # Parse solver-specific attributes imported from USD
        kamino_attrs = getattr(model, "kamino", None)
        if kamino_attrs is not None:
            if hasattr(kamino_attrs, "constraints_use_preconditioning"):
                cfg.preconditioning = bool(kamino_attrs.constraints_use_preconditioning.numpy()[0])

        # Return the fully constructed config with configurations
        # parsed from the model's custom attributes if available,
        # otherwise using defaults or provided kwargs.
        return cfg

    @override
    def validate(self) -> None:
        """
        Validates the current values held by the :class:`ConstrainedDynamicsConfig` instance.
        """
        # Import here to avoid module-level imports and circular dependencies
        from ._src.linalg import LinearSolverNameToType  # noqa: PLC0415

        # Ensure that the linear solver type is a valid option
        supported_linear_solver_types = LinearSolverNameToType.keys()
        if self.linear_solver_type not in supported_linear_solver_types:
            raise ValueError(
                f"Invalid linear_solver_type: {self.linear_solver_type}. "
                f"Must be one of {supported_linear_solver_types}."
            )

    @override
    def __post_init__(self):
        """Post-initialization to validate configurations."""
        self.validate()


@dataclass
class PADMMSolverConfig:
    """
    A container to hold configurations for the PADMM forward dynamics solver.
    """

    primal_tolerance: float = 1e-6
    """
    The target tolerance on the total primal residual `r_primal`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    dual_tolerance: float = 1e-6
    """
    The target tolerance on the total dual residual `r_dual`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    compl_tolerance: float = 1e-6
    """
    The target tolerance on the total complementarity residual `r_compl`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    restart_tolerance: float = 0.999
    """
    The tolerance on the total combined primal-dual residual `r_comb`,
    for determining when gradient acceleration should be restarted.\n
    Must be greater than zero. Defaults to `0.999`.
    """

    eta: float = 1e-5
    """
    The proximal regularization parameter.\n
    Must be greater than zero. Defaults to `1e-5`.
    """

    rho_0: float = 1.0
    """
    The initial value of the ALM penalty parameter.\n
    Must be greater than zero. Defaults to `1.0`.
    """

    rho_min: float = 1e-5
    """
    The lower-bound applied to the ALM penalty parameter.\n
    Used to ensure numerical stability when adaptive penalty updates are used.\n
    Must be greater than zero. Defaults to `1e-5`.
    """

    a_0: float = 1.0
    """
    The initial value of the acceleration parameter.\n
    Must be greater than zero. Defaults to `1.0`.
    """

    alpha: float = 10.0
    """
    The primal-dual residual threshold used to determine when penalty updates are needed.
    Must be greater than one. Defaults to `10.0`.
    """

    tau: float = 1.5
    """
    The factor by which the ALM penalty is increased/decreased when
    the primal-dual residual ratios exceed the threshold `alpha`.\n
    Must be greater than `1.0`. Defaults to `1.5`.
    """

    max_iterations: int = 200
    """
    The maximum number of solver iterations.\n
    Must be greater than zero. Defaults to `200`.
    """

    penalty_update_freq: int = 1
    """
    The permitted frequency of penalty updates.\n
    If zero, no updates are performed. Otherwise, updates are performed every
    `penalty_update_freq` iterations. Defaults to `1`.
    """

    penalty_update_method: Literal["fixed", "balanced"] = "fixed"
    """
    The penalty update method used to adapt the penalty parameter.\n
    Defaults to `fixed`. See :class:`PADMMPenaltyUpdate` for details.
    """

    linear_solver_tolerance: float = 0.0
    """
    The default absolute tolerance for the iterative linear solver.\n
    When zero, the iterative solver's own tolerance is left unchanged.\n
    When positive, the iterative solver's atol is initialized
    to this value at the start of each ADMM solve.\n
    Must be non-negative. Defaults to `0.0`.
    """

    linear_solver_tolerance_ratio: float = 0.0
    """
    The ratio used to adapt the iterative linear solver tolerance from the ADMM primal residual.\n
    When zero, the linear solver tolerance is not adapted (fixed tolerance).\n
    When positive, the linear solver absolute tolerance is
    set to `ratio * ||r_primal||_2` at each ADMM iteration.\n
    Must be non-negative. Defaults to `0.0`.
    """

    use_acceleration: bool = True
    """
    Enables Nesterov-type acceleration, i.e. use APADMM instead of standard PADMM.\n
    Defaults to `True`.
    """

    use_graph_conditionals: bool = True
    """
    Enables use of CUDA graph conditional nodes in iterative solvers.\n
    If `False`, replaces `wp.capture_while` with unrolled for-loops over max iterations.\n
    Defaults to `True`.
    """

    warmstart_mode: Literal["none", "internal", "containers"] = "containers"
    """
    Warmstart mode to be used for the dynamics solver.\n
    See :class:`PADMMWarmStartMode` for the available options.\n
    Defaults to `containers` to warmstart from the solver data containers.
    """

    contact_warmstart_method: Literal[
        "key_and_position",
        "geom_pair_net_force",
        "geom_pair_net_wrench",
        "key_and_position_with_net_force_backup",
        "key_and_position_with_net_wrench_backup",
    ] = "key_and_position"
    """
    Method to be used for warm-starting contacts.\n
    See :class:`WarmstarterContacts.Method` for available options.\n
    Defaults to `key_and_position`.
    """

    @override
    @staticmethod
    def register_custom_attributes(builder: ModelBuilder) -> None:
        """
        Registers custom attributes for the PADMM solver configurations with the given builder.

        Note: Currently, not all configurations are registered as custom attributes,
        as only those supported by the Kamino USD scene API have been included. More
        will be added in the future as latter is being developed.

        Args:
            builder: The model builder instance with which to register the custom attributes.
        """
        # Import here to avoid module-level imports and circular dependencies
        from ._src.solvers.padmm import PADMMWarmStartMode  # noqa: PLC0415

        # Separately register `newton:maxSolverIterations` from
        # `KaminoSceneAPI` so we have access to it through the model.
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="max_solver_iterations",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="kamino",
                usd_attribute_name="newton:maxSolverIterations",
            )
        )

        # Register KaminoSceneAPI attributes so the USD importer will store them on the model
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="padmm_primal_tolerance",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-6,
                namespace="kamino",
                usd_attribute_name="newton:kamino:padmm:primalTolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="padmm_dual_tolerance",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-6,
                namespace="kamino",
                usd_attribute_name="newton:kamino:padmm:dualTolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="padmm_complementarity_tolerance",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-6,
                namespace="kamino",
                usd_attribute_name="newton:kamino:padmm:complementarityTolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="padmm_use_acceleration",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=True,
                namespace="kamino",
                usd_attribute_name="newton:kamino:padmm:useAcceleration",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="padmm_warmstarting",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=str,
                default="containers",
                namespace="kamino",
                usd_attribute_name="newton:kamino:padmm:warmstarting",
                usd_value_transformer=PADMMWarmStartMode.parse_usd_attribute,
            )
        )

    @override
    @staticmethod
    def from_model(model: Model, **kwargs: dict[str, Any]) -> PADMMSolverConfig:
        """
        Creates a :class:`PADMMSolverConfig` by attempting to
        parse custom attributes from a :class:`Model` if available.

        Args:
            model: The Newton model from which to parse configurations.
        """
        cfg = PADMMSolverConfig(**kwargs)

        # Parse solver-specific attributes imported from USD
        kamino_attrs = getattr(model, "kamino", None)
        if kamino_attrs is not None:
            if hasattr(kamino_attrs, "max_solver_iterations"):
                max_iterations = kamino_attrs.max_solver_iterations.numpy()[0]
                if max_iterations >= 0:
                    cfg.max_iterations = max_iterations
            if hasattr(kamino_attrs, "padmm_primal_tolerance"):
                cfg.primal_tolerance = float(kamino_attrs.padmm_primal_tolerance.numpy()[0])
            if hasattr(kamino_attrs, "padmm_dual_tolerance"):
                cfg.dual_tolerance = float(kamino_attrs.padmm_dual_tolerance.numpy()[0])
            if hasattr(kamino_attrs, "padmm_complementarity_tolerance"):
                cfg.compl_tolerance = float(kamino_attrs.padmm_complementarity_tolerance.numpy()[0])
            if hasattr(kamino_attrs, "padmm_warmstarting"):
                cfg.warmstart_mode = kamino_attrs.padmm_warmstarting[0]
            if hasattr(kamino_attrs, "padmm_use_acceleration"):
                cfg.use_acceleration = bool(kamino_attrs.padmm_use_acceleration.numpy()[0])

        # Return the fully constructed config with configurations
        # parsed from the model's custom attributes if available,
        # otherwise using defaults or provided kwargs.
        return cfg

    @override
    def validate(self) -> None:
        """
        Validates the current values held by the :class:`PADMMSolverConfig` instance.
        """
        # Import here to avoid module-level imports and circular dependencies
        from ._src.solvers.padmm import PADMMPenaltyUpdate, PADMMWarmStartMode  # noqa: PLC0415
        from ._src.solvers.warmstart import WarmstarterContacts  # noqa: PLC0415

        # Ensure that the scalar parameters are within valid ranges
        if self.primal_tolerance < 0.0:
            raise ValueError(f"Invalid primal tolerance: {self.primal_tolerance}. Must be non-negative.")
        if self.dual_tolerance < 0.0:
            raise ValueError(f"Invalid dual tolerance: {self.dual_tolerance}. Must be non-negative.")
        if self.compl_tolerance < 0.0:
            raise ValueError(f"Invalid complementarity tolerance: {self.compl_tolerance}. Must be non-negative.")
        if not (0.0 <= self.restart_tolerance < 1.0):
            raise ValueError(f"Invalid restart tolerance: {self.restart_tolerance}. Must be in the range [0.0, 1.0).")
        if self.eta <= 0.0:
            raise ValueError(f"Invalid proximal parameter: {self.eta}. Must be greater than zero.")
        if self.rho_0 <= 0.0:
            raise ValueError(f"Invalid initial ALM penalty: {self.rho_0}. Must be greater than zero.")
        if self.rho_min <= 0.0:
            raise ValueError(f"Invalid minimum ALM penalty: {self.rho_min}. Must be greater than zero.")
        if self.a_0 <= 0.0:
            raise ValueError(f"Invalid initial acceleration parameter: {self.a_0}. Must be greater than zero.")
        if self.alpha <= 1.0:
            raise ValueError(f"Invalid penalty threshold: {self.alpha}. Must be greater than one.")
        if self.tau <= 1.0:
            raise ValueError(f"Invalid penalty increment factor: {self.tau}. Must be greater than one.")
        if self.max_iterations <= 0:
            raise ValueError(f"Invalid maximum iterations: {self.max_iterations}. Must be a positive integer.")
        if self.penalty_update_freq < 0:
            raise ValueError(f"Invalid penalty update frequency: {self.penalty_update_freq}. Must be non-negative.")
        if self.linear_solver_tolerance < 0.0:
            raise ValueError(f"Invalid linear solver tolerance: {self.linear_solver_tolerance}. Must be non-negative.")
        if self.linear_solver_tolerance_ratio < 0.0:
            raise ValueError(
                f"Invalid linear solver tolerance ratio: {self.linear_solver_tolerance_ratio}. Must be non-negative."
            )

        # Ensure that the enum-valued parameters are valid options
        # Conversion to enum-type configs will raise an error
        # if the corresponding input string is invalid.
        PADMMPenaltyUpdate.from_string(self.penalty_update_method)
        PADMMWarmStartMode.from_string(self.warmstart_mode)
        WarmstarterContacts.Method.from_string(self.contact_warmstart_method)

    @override
    def __post_init__(self):
        """Post-initialization to validate configurations."""
        self.validate()


@dataclass
class ForwardKinematicsSolverConfig:
    """
    A container to hold configurations for the Gauss-Newton forward kinematics solver used for state resets.
    """

    preconditioner: Literal["none", "jacobi_diagonal", "jacobi_block_diagonal"] = "jacobi_block_diagonal"
    """
    Preconditioner to use for the Conjugate Gradient solver if sparsity is enabled
    Changing this setting after the solver's initialization leads to undefined behavior.
    Defaults to `jacobi_block_diagonal`.
    """

    max_newton_iterations: int = 30
    """
    Maximal number of Gauss-Newton iterations.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `30`.
    """

    max_line_search_iterations: int = 20
    """
    Maximal line search iterations in the inner loop.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `20`.
    """

    tolerance: float = 1e-6
    """
    Maximal absolute kinematic constraint value that is acceptable at the solution.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `1e-6`.
    """

    TILE_SIZE_CTS: int = 8
    """
    Tile size for kernels along the dimension of kinematic constraints.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `8`.
    """

    TILE_SIZE_VRS: int = 8
    """
    Tile size for kernels along the dimension of rigid body pose variables.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `8`.
    """

    use_sparsity: bool = False
    """
    Whether to use sparse Jacobian and solver; otherwise, dense versions are used.
    Changes to this setting after the solver's initialization lead to undefined behavior.
    Defaults to `False`.
    """

    use_adaptive_cg_tolerance: bool = True
    """
    Whether to use an adaptive tolerance strategy for the Conjugate Gradient solver if sparsity
    is enabled, which reduces the number of CG iterations in most cases.
    Changes to this setting after graph capture will have no effect.
    Defaults to `True`.
    """

    reset_state: bool = True
    """
    Whether to reset the state to initial states, to use as initial guess.
    Changes to this setting after graph capture will have no effect.
    Defaults to `True`.
    """

    @override
    @staticmethod
    def register_custom_attributes(builder: ModelBuilder) -> None:
        """
        Registers custom attributes for the FK solver configurations with the given builder.

        Note: Currently, this class does not have any custom attributes registered,
        as only those supported by the Kamino USD scene API have been included. More
        will be added in the future as latter is being developed.

        Args:
            builder: The model builder instance with which to register the custom attributes.
        """
        pass  # TODO: Add custom attributes for the FK solver when supported by the Kamino USD scene API

    @override
    @staticmethod
    def from_model(model: Model, **kwargs: dict[str, Any]) -> ForwardKinematicsSolverConfig:
        """
        Creates a :class:`ForwardKinematicsSolverConfig` by attempting
        to parse custom attributes from a :class:`Model` if available.

        Args:
            model: The Newton model from which to parse configurations.
        """
        cfg = ForwardKinematicsSolverConfig(**kwargs)

        # TODO: Implement these

        # Return the fully constructed config with configurations
        # parsed from the model's custom attributes if available,
        # otherwise using defaults or provided kwargs.
        return cfg

    @override
    def validate(self) -> None:
        """
        Validates the current values held by the :class:`ForwardKinematicsSolverConfig` instance.
        """
        # Import here to avoid module-level imports and circular dependencies
        from ._src.solvers.fk import ForwardKinematicsSolver  # noqa: PLC0415

        # Ensure that the enum-valued parameters are valid options
        ForwardKinematicsSolver.PreconditionerType.from_string(self.preconditioner)

        # Ensure that the integer and float parameters are within valid ranges
        if self.max_newton_iterations <= 0:
            raise ValueError("`max_newton_iterations` must be positive.")
        if self.max_line_search_iterations <= 0:
            raise ValueError("`max_line_search_iterations` must be positive.")
        if self.tolerance <= 0.0:
            raise ValueError("`tolerance` must be positive.")
        if self.TILE_SIZE_CTS <= 0:
            raise ValueError("`TILE_SIZE_CTS` must be positive.")
        if self.TILE_SIZE_VRS <= 0:
            raise ValueError("`TILE_SIZE_VRS` must be positive.")

    @override
    def __post_init__(self):
        """Post-initialization to validate configurations."""
        self.validate()
