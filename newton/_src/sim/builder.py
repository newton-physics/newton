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

"""A module for building Newton models."""

from __future__ import annotations

import copy
import ctypes
import math
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import warp as wp

from ..core.spatial import quat_between_vectors_robust
from ..core.types import (
    MAXVAL,
    Axis,
    AxisType,
    Devicelike,
    Mat33,
    Quat,
    Sequence,
    Transform,
    Vec3,
    Vec4,
    axis_to_vec3,
    flag_to_int,
    nparray,
)
from ..geometry import (
    SDF,
    GeoType,
    Mesh,
    ParticleFlags,
    ShapeFlags,
    compute_shape_inertia,
    compute_shape_radius,
    transform_inertia,
)
from ..geometry.inertia import validate_and_correct_inertia_kernel, verify_and_correct_inertia
from ..geometry.utils import RemeshingMethod, compute_inertia_obb, remesh_mesh
from ..usd.schema_resolver import SchemaResolver
from ..utils import compute_world_offsets
from ..utils.mesh import MeshAdjacency
from .graph_coloring import (
    ColoringAlgorithm,
    color_graph,
    color_rigid_bodies,
    combine_independent_particle_coloring,
    construct_particle_graph,
)
from .joints import (
    ActuatorMode,
    EqType,
    JointType,
)
from .model import Model


class ModelBuilder:
    """A helper class for building simulation models at runtime.

    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
    represents the scene using standard Python data structures like lists,
    which are convenient but unsuitable for efficient simulation.
    Call :meth:`finalize` to construct a simulation-ready Model.

    Example
    -------

    .. testcode::

        import newton
        from newton.solvers import SolverXPBD

        builder = newton.ModelBuilder()

        # anchor point (zero mass)
        builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

        # build chain
        for i in range(1, 10):
            builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
            builder.add_spring(i - 1, i, 1.0e3, 0.0, 0)

        # create model
        model = builder.finalize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        solver = SolverXPBD(model)

        for i in range(10):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt=1.0 / 60.0)
            state_0, state_1 = state_1, state_0

    World Grouping
    --------------------

    ModelBuilder supports world grouping to organize entities for multi-world simulations.
    Each entity (particle, body, shape, joint, articulation) has an associated world index:

    - Index -1: Global entities shared across all worlds (e.g., ground plane)
    - Index 0, 1, 2, ...: World-specific entities

    There are two ways to assign world indices:

    1. **Direct entity creation**: Entities inherit the builder's `current_world` value::

           builder = ModelBuilder()
           builder.current_world = -1  # Following entities will be global
           builder.add_ground_plane()
           builder.current_world = 0  # Following entities will be in world 0
           builder.add_body(...)

    2. **Using add_world()**: ALL entities from the sub-builder are assigned to a new world::

           robot = ModelBuilder()
           robot.add_body(...)  # World assignments here will be overridden

           main = ModelBuilder()
           main.add_world(robot)  # All robot entities -> world 0
           main.add_world(robot)  # All robot entities -> world 1

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    @dataclass
    class ShapeConfig:
        """
        Represents the properties of a collision shape used in simulation.
        """

        density: float = 1000.0
        """The density of the shape material."""
        ke: float = 2.5e3
        """The contact elastic stiffness. Used by SemiImplicit, Featherstone, MuJoCo."""
        kd: float = 100.0
        """The contact damping coefficient. Used by SemiImplicit, Featherstone, MuJoCo."""
        kf: float = 1000.0
        """The friction damping coefficient. Used by SemiImplicit, Featherstone."""
        ka: float = 0.0
        """The contact adhesion distance. Used by SemiImplicit, Featherstone."""
        mu: float = 0.5
        """The coefficient of friction. Used by all solvers."""
        restitution: float = 0.0
        """The coefficient of restitution. Used by XPBD. To take effect, enable restitution in solver constructor via ``enable_restitution=True``."""
        torsional_friction: float = 0.25
        """The coefficient of torsional friction (resistance to spinning at contact point). Used by XPBD, MuJoCo."""
        rolling_friction: float = 0.0005
        """The coefficient of rolling friction (resistance to rolling motion). Used by XPBD, MuJoCo."""
        thickness: float = 1e-5
        """Outward offset from the shape's surface for collision detection.
        Extends the effective collision surface outward by this amount. When two shapes collide,
        their thicknesses are summed (thickness_a + thickness_b) to determine the total separation."""
        contact_margin: float | None = None
        """The contact margin for collision detection. If None, uses builder.rigid_contact_margin as default.
        AABBs are expanded by this value for broad phase detection. Must be >= thickness to ensure
        collisions are not missed when thickened surfaces approach each other."""
        is_solid: bool = True
        """Indicates whether the shape is solid or hollow. Defaults to True."""
        collision_group: int = 1
        """The collision group ID for the shape. Defaults to 1 (default group). Set to 0 to disable collisions for this shape."""
        collision_filter_parent: bool = True
        """Whether to inherit collision filtering from the parent. Defaults to True."""
        has_shape_collision: bool = True
        """Whether the shape can collide with other shapes. Defaults to True."""
        has_particle_collision: bool = True
        """Whether the shape can collide with particles. Defaults to True."""
        is_visible: bool = True
        """Indicates whether the shape is visible in the simulation. Defaults to True."""
        is_site: bool = False
        """Indicates whether the shape is a site (non-colliding reference point). Directly setting this to True will NOT enforce site invariants. Use `mark_as_site()` or set via the `flags` property to ensure invariants. Defaults to False."""
        sdf_narrow_band_range: tuple[float, float] = (-0.1, 0.1)
        """The narrow band distance range (inner, outer) for SDF computation. Only used for mesh shapes when SDF is enabled."""
        sdf_target_voxel_size: float | None = None
        """Target voxel size for sparse SDF grid.
        If provided, enables SDF generation and takes precedence over sdf_max_resolution.
        Requires GPU since wp.Volume only supports CUDA. Only used for mesh shapes."""
        sdf_max_resolution: int | None = None
        """Maximum dimension for sparse SDF grid (must be divisible by 8).
        If provided (and sdf_target_voxel_size is None), enables SDF-based mesh-mesh collision.
        Set to None (default) to disable SDF generation for this shape (uses BVH-based collision for mesh-mesh instead).
        Requires GPU since wp.Volume only supports CUDA. Only used for mesh shapes."""
        is_hydroelastic: bool = False
        """Whether the shape collides using SDF-based hydroelastics. For hydroelastic collisions, both participating shapes must have is_hydroelastic set to True. Defaults to False.

        .. note::
            Hydroelastic collision handling only works with volumetric shapes and in particular will not work for shapes like flat meshes or cloth.
            This flag will be automatically set to False for planes and heightfields in :meth:`ModelBuilder.add_shape`.
        """
        k_hydro: float = 1.0e10
        """Contact stiffness coefficient for hydroelastic collisions. Used by MuJoCo, Featherstone, SemiImplicit when is_hydroelastic is True.

        .. note::
            For MuJoCo, stiffness values will internally be scaled by masses.
            Users should choose k_hydro to match their desired force-to-penetration ratio.
        """

        def validate(self) -> None:
            """Validate ShapeConfig parameters."""
            if self.sdf_max_resolution is not None and self.sdf_max_resolution % 8 != 0:
                raise ValueError(
                    f"sdf_max_resolution must be divisible by 8 (got {self.sdf_max_resolution}). "
                    "This is required because SDF volumes are allocated in 8x8x8 tiles."
                )
            if (
                self.is_hydroelastic
                and self.has_shape_collision
                and self.sdf_max_resolution is None
                and self.sdf_target_voxel_size is None
            ):
                raise ValueError(
                    "Hydroelastic shapes require an SDF. Set either sdf_max_resolution or sdf_target_voxel_size."
                )

        def mark_as_site(self) -> None:
            """Marks this shape as a site and enforces all site invariants.

            Sets:
            - is_site = True
            - has_shape_collision = False
            - has_particle_collision = False
            - density = 0.0
            - collision_group = 0
            """
            self.is_site = True
            self.has_shape_collision = False
            self.has_particle_collision = False
            self.density = 0.0
            self.collision_group = 0

        @property
        def flags(self) -> int:
            """Returns the flags for the shape."""

            shape_flags = ShapeFlags.VISIBLE if self.is_visible else 0
            shape_flags |= ShapeFlags.COLLIDE_SHAPES if self.has_shape_collision else 0
            shape_flags |= ShapeFlags.COLLIDE_PARTICLES if self.has_particle_collision else 0
            shape_flags |= ShapeFlags.SITE if self.is_site else 0
            shape_flags |= ShapeFlags.HYDROELASTIC if self.is_hydroelastic else 0
            return shape_flags

        @flags.setter
        def flags(self, value: int):
            """Sets the flags for the shape."""

            self.is_visible = bool(value & ShapeFlags.VISIBLE)
            self.is_hydroelastic = bool(value & ShapeFlags.HYDROELASTIC)

            # Check if SITE flag is being set
            is_site_flag = bool(value & ShapeFlags.SITE)

            if is_site_flag:
                # Use mark_as_site() to enforce invariants
                self.mark_as_site()
                # Collision flags will be cleared by mark_as_site()
            else:
                # SITE flag is being cleared - restore non-site defaults
                defaults = self.__class__()
                self.is_site = False
                self.density = defaults.density
                self.collision_group = defaults.collision_group
                self.has_shape_collision = bool(value & ShapeFlags.COLLIDE_SHAPES)
                self.has_particle_collision = bool(value & ShapeFlags.COLLIDE_PARTICLES)

        def copy(self) -> ModelBuilder.ShapeConfig:
            return copy.copy(self)

    class JointDofConfig:
        """
        Describes a joint axis (a single degree of freedom) that can have limits and be driven towards a target.
        """

        def __init__(
            self,
            axis: AxisType | Vec3 = Axis.X,
            limit_lower: float = -MAXVAL,
            limit_upper: float = MAXVAL,
            limit_ke: float = 1e4,
            limit_kd: float = 1e1,
            target_pos: float = 0.0,
            target_vel: float = 0.0,
            target_ke: float = 0.0,
            target_kd: float = 0.0,
            armature: float = 1e-2,
            effort_limit: float = 1e6,
            velocity_limit: float = 1e6,
            friction: float = 0.0,
            actuator_mode: ActuatorMode | None = None,
        ):
            self.axis = wp.normalize(axis_to_vec3(axis))
            """The 3D joint axis in the joint parent anchor frame."""
            self.limit_lower = limit_lower
            """The lower position limit of the joint axis. Defaults to -MAXVAL (unlimited)."""
            self.limit_upper = limit_upper
            """The upper position limit of the joint axis. Defaults to MAXVAL (unlimited)."""
            self.limit_ke = limit_ke
            """The elastic stiffness of the joint axis limits. Defaults to 1e4."""
            self.limit_kd = limit_kd
            """The damping stiffness of the joint axis limits. Defaults to 1e1."""
            self.target_pos = target_pos
            """The target position of the joint axis.
            If the initial `target_pos` is outside the limits,
            it defaults to the midpoint of `limit_lower` and `limit_upper`. Otherwise, defaults to 0.0."""
            self.target_vel = target_vel
            """The target velocity of the joint axis."""
            self.target_ke = target_ke
            """The proportional gain of the target drive PD controller. Defaults to 0.0."""
            self.target_kd = target_kd
            """The derivative gain of the target drive PD controller. Defaults to 0.0."""
            self.armature = armature
            """Artificial inertia added around the joint axis. Defaults to 1e-2."""
            self.effort_limit = effort_limit
            """Maximum effort (force or torque) the joint axis can exert. Defaults to 1e6."""
            self.velocity_limit = velocity_limit
            """Maximum velocity the joint axis can achieve. Defaults to 1e6."""
            self.friction = friction
            """Friction coefficient for the joint axis. Defaults to 0.0."""
            self.actuator_mode = actuator_mode
            """Actuator mode for this DOF. Determines which actuators are installed (see :class:`ActuatorMode`).
            If None, the mode is inferred from gains and targets."""

            if self.target_pos > self.limit_upper or self.target_pos < self.limit_lower:
                self.target_pos = 0.5 * (self.limit_lower + self.limit_upper)

        @classmethod
        def create_unlimited(cls, axis: AxisType | Vec3) -> ModelBuilder.JointDofConfig:
            """Creates a JointDofConfig with no limits."""
            return ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=-MAXVAL,
                limit_upper=MAXVAL,
                target_pos=0.0,
                target_vel=0.0,
                target_ke=0.0,
                target_kd=0.0,
                armature=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
            )

    @dataclass
    class CustomAttribute:
        """
        Represents a custom attribute definition for the ModelBuilder.
        This is used to define custom attributes that are not part of the standard ModelBuilder API.
        Custom attributes can be defined for the :class:`~newton.Model`, :class:`~newton.State`, :class:`~newton.Control`, or :class:`~newton.Contacts` objects, depending on the :class:`Model.AttributeAssignment` category.
        Custom attributes must be declared before use via the :meth:`newton.ModelBuilder.add_custom_attribute` method.

        See :ref:`custom_attributes` for more information.
        """

        name: str
        """Variable name to expose on the Model. Must be a valid Python identifier."""

        dtype: type
        """Warp dtype (e.g., wp.float32, wp.int32, wp.bool, wp.vec3) that is compatible with Warp arrays,
        or ``str`` for string attributes that remain as Python lists."""

        frequency: Model.AttributeFrequency | str
        """Frequency category that determines how the attribute is indexed in the Model.

        Can be either:
            - A :class:`Model.AttributeFrequency` enum value for built-in frequencies (BODY, SHAPE, JOINT, etc.)
              Uses dict-based storage where keys are entity indices, allowing sparse assignment.
            - A string for custom frequencies (e.g., ``"mujoco:pair"``). Uses list-based storage for
              sequential data appended via :meth:`add_custom_values`. All attributes sharing the same
              custom frequency must have the same count, validated at finalize time."""

        assignment: Model.AttributeAssignment = Model.AttributeAssignment.MODEL
        """Assignment category (see :class:`Model.AttributeAssignment`), defaults to :attr:`Model.AttributeAssignment.MODEL`"""

        namespace: str | None = None
        """Namespace for the attribute. If None, the attribute is added directly to the assigned object without a namespace."""

        references: str | None = None
        """For attributes containing entity indices, specifies how values are transformed during add_world/add_builder merging.

        Built-in entity types (values are offset by entity count):
            - ``"body"``, ``"shape"``, ``"joint"``, ``"joint_dof"``, ``"joint_coord"``, ``"articulation"``, ``"equality_constraint"``,
              ``"constraint_mimic"``, ``"particle"``, ``"edge"``, ``"triangle"``, ``"tetrahedron"``, ``"spring"``

        Special handling:
            - ``"world"``: Values are replaced with ``current_world`` (not offset)

        Custom frequencies (values are offset by that frequency's count):
            - Any custom frequency string, e.g., ``"mujoco:pair"``
        """

        default: Any = None
        """Default value for the attribute. If None, the default value is determined based on the dtype."""

        values: dict[int, Any] | list[Any] | None = None
        """Storage for specific values (overrides).

        For enum frequencies (BODY, SHAPE, etc.): dict[int, Any] mapping entity indices to values.
        For string frequencies ("mujoco:pair", etc.): list[Any] for sequential custom data.

        If None, the attribute is not initialized with any values. Values can be assigned in subsequent
        ``ModelBuilder.add_*(..., custom_attributes={...})`` method calls for specific entities after
        the CustomAttribute has been added through the :meth:`ModelBuilder.add_custom_attribute` method."""

        usd_attribute_name: str | None = None
        """Name of the corresponding USD attribute. If None, the USD attribute name ``"newton:<namespace>:<name>"`` is used."""

        mjcf_attribute_name: str | None = None
        """Name of the attribute in the MJCF definition. If None, the attribute name is used."""

        urdf_attribute_name: str | None = None
        """Name of the attribute in the URDF definition. If None, the attribute name is used."""

        usd_value_transformer: Callable[[Any, dict[str, Any] | None], Any] | None = None
        """Transformer function that converts a USD attribute value to a valid Warp dtype. If undefined, the generic converter from :func:`newton.usd.convert_warp_value` is used. Receives an optional context dict with parsing-time information."""

        mjcf_value_transformer: Callable[[str, dict[str, Any] | None], Any] | None = None
        """Transformer function that converts a MJCF attribute value string to a valid Warp dtype. If undefined, the generic converter from :func:`newton.utils.parse_warp_value_from_string` is used. Receives an optional context dict with parsing-time information (e.g., use_degrees, joint_type)."""

        urdf_value_transformer: Callable[[str, dict[str, Any] | None], Any] | None = None
        """Transformer function that converts a URDF attribute value string to a valid Warp dtype. If undefined, the generic converter from :func:`newton.utils.parse_warp_value_from_string` is used. Receives an optional context dict with parsing-time information."""

        def __post_init__(self):
            """Initialize default values and validate dtype compatibility."""
            # Allow str dtype for string attributes (stored as Python lists, not warp arrays)
            if self.dtype is not str:
                # ensure dtype is a valid Warp dtype
                try:
                    _size = wp.types.type_size_in_bytes(self.dtype)
                except TypeError as e:
                    raise ValueError(f"Invalid dtype: {self.dtype}. Must be a valid Warp dtype or str.") from e

            # Set dtype-specific default value if none was provided
            if self.default is None:
                self.default = self._default_for_dtype(self.dtype)

            # Initialize values with correct container type based on frequency
            if self.values is None:
                self.values = self._create_empty_values_container()
            if self.usd_attribute_name is None:
                self.usd_attribute_name = f"newton:{self.key}"
            if self.mjcf_attribute_name is None:
                self.mjcf_attribute_name = self.name
            if self.urdf_attribute_name is None:
                self.urdf_attribute_name = self.name

        @staticmethod
        def _default_for_dtype(dtype: object) -> Any:
            """Get default value for dtype when not specified."""
            # string type gets empty string
            if dtype is str:
                return ""
            # quaternions get identity quaternion
            if wp.types.type_is_quaternion(dtype):
                return wp.quat_identity(dtype._wp_scalar_type_)
            if dtype is wp.bool or dtype is bool:
                return False
            # vectors, matrices, scalars
            return dtype(0)

        @property
        def key(self) -> str:
            """Return the full name of the attribute, formatted as "namespace:name" or "name" if no namespace is specified."""
            return f"{self.namespace}:{self.name}" if self.namespace else self.name

        @property
        def frequency_key(self) -> Model.AttributeFrequency | str:
            """Return the resolved frequency, with namespace prepended for custom string frequencies.

            For string frequencies: returns "namespace:frequency" if namespace is set, otherwise just "frequency".
            For enum frequencies: returns the enum value unchanged.
            """
            if isinstance(self.frequency, str):
                return (
                    f"{self.namespace}:{self.frequency}"
                    if self.namespace and ":" not in self.frequency
                    else self.frequency
                )
            return self.frequency

        @property
        def is_custom_frequency(self) -> bool:
            """Check if this attribute uses a custom (string) frequency.

            Returns:
                True if the frequency is a string (custom frequency), False if it's a
                Model.AttributeFrequency enum (built-in frequency like BODY, SHAPE, etc.).
            """
            return isinstance(self.frequency, str)

        def _create_empty_values_container(self) -> list | dict:
            """Create appropriate empty container based on frequency type."""
            return [] if self.is_custom_frequency else {}

        def _get_values_count(self) -> int:
            """Get current count of values in this attribute."""
            if self.values is None:
                return 0
            return len(self.values)

        def build_array(
            self, count: int, device: Devicelike | None = None, requires_grad: bool = False
        ) -> wp.array | list:
            """Build wp.array (or list for string dtype) from count, dtype, default and overrides.

            For string dtype, returns a Python list[str] instead of a Warp array.
            """
            if self.values is None or len(self.values) == 0:
                # No values provided, use default for all
                arr = [self.default] * count
            elif self.is_custom_frequency:
                # Custom frequency: vals is a list, replace None with defaults and pad/truncate as needed
                arr = [val if val is not None else self.default for val in self.values]
                arr = arr + [self.default] * max(0, count - len(arr))
                arr = arr[:count]  # Truncate if needed
            else:
                # Enum frequency: vals is a dict, use get() to fill gaps with defaults
                arr = [self.values.get(i, self.default) for i in range(count)]

            # String dtype: return as Python list instead of warp array
            if self.dtype is str:
                return arr

            return wp.array(arr, dtype=self.dtype, requires_grad=requires_grad, device=device)

    def __init__(self, up_axis: AxisType = Axis.Z, gravity: float = -9.81):
        """
        Initializes a new ModelBuilder instance for constructing simulation models.

        Args:
            up_axis (AxisType, optional): The axis to use as the "up" direction in the simulation.
                Defaults to Axis.Z.
            gravity (float, optional): The magnitude of gravity to apply along the up axis.
                Defaults to -9.81.
        """
        self.num_worlds = 0

        # region defaults
        self.default_shape_cfg = ModelBuilder.ShapeConfig()
        self.default_joint_cfg = ModelBuilder.JointDofConfig()

        # Default particle settings
        self.default_particle_radius = 0.1

        # Default triangle soft mesh settings
        self.default_tri_ke = 100.0
        self.default_tri_ka = 100.0
        self.default_tri_kd = 10.0
        self.default_tri_drag = 0.0
        self.default_tri_lift = 0.0

        # Default distance constraint properties
        self.default_spring_ke = 100.0
        self.default_spring_kd = 0.0

        # Default edge bending properties
        self.default_edge_ke = 100.0
        self.default_edge_kd = 0.0

        # Default body settings
        self.default_body_armature = 0.0
        # endregion

        # region compiler settings (similar to MuJoCo)
        self.balance_inertia: bool = True
        """Whether to automatically correct rigid body inertia tensors that violate the triangle inequality.
        When True, adds a scalar multiple of the identity matrix to preserve rotation structure while
        ensuring physical validity (I1 + I2 >= I3 for principal moments). Default: True."""

        self.bound_mass: float | None = None
        """Minimum allowed mass value for rigid bodies. If set, any body mass below this value will be
        clamped to this minimum. Set to None to disable mass clamping. Default: None."""

        self.bound_inertia: float | None = None
        """Minimum allowed eigenvalue for rigid body inertia tensors. If set, ensures all principal
        moments of inertia are at least this value. Set to None to disable inertia eigenvalue
        clamping. Default: None."""

        self.validate_inertia_detailed: bool = False
        """Whether to use detailed (slower) inertia validation that provides per-body warnings.
        When False, uses a fast GPU kernel that reports only the total number of corrected bodies
        and directly assigns the corrected arrays to the Model (ModelBuilder state is not updated).
        When True, uses a CPU implementation that reports specific issues for each body and updates
        the ModelBuilder's internal state.
        Default: False."""

        # endregion

        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []
        self.particle_radius = []
        self.particle_flags = []
        self.particle_max_velocity = 1e5
        self.particle_color_groups: list[nparray] = []
        self.particle_world = []  # world index for each particle

        # shapes (each shape has an entry in these arrays)
        self.shape_key = []  # shape keys
        # transform from shape to body
        self.shape_transform = []
        # maps from shape index to body index
        self.shape_body = []
        self.shape_flags = []
        self.shape_type = []
        self.shape_scale = []
        self.shape_source = []
        self.shape_is_solid = []
        self.shape_thickness = []
        self.shape_material_ke = []
        self.shape_material_kd = []
        self.shape_material_kf = []
        self.shape_material_ka = []
        self.shape_material_mu = []
        self.shape_material_restitution = []
        self.shape_material_torsional_friction = []
        self.shape_material_rolling_friction = []
        self.shape_material_k_hydro = []
        self.shape_contact_margin = []
        # collision groups within collisions are handled
        self.shape_collision_group = []
        # radius to use for broadphase collision checking
        self.shape_collision_radius = []
        # world index for each shape
        self.shape_world = []
        # SDF parameters per shape
        self.shape_sdf_narrow_band_range = []
        self.shape_sdf_target_voxel_size = []
        self.shape_sdf_max_resolution = []

        # Mesh SDF storage (volumes kept for reference counting, SDFData array created at finalize)

        # filtering to ignore certain collision pairs
        self.shape_collision_filter_pairs: list[tuple[int, int]] = []

        self._requested_contact_attributes: set[str] = set()
        self._requested_state_attributes: set[str] = set()

        # springs
        self.spring_indices = []
        self.spring_rest_length = []
        self.spring_stiffness = []
        self.spring_damping = []
        self.spring_control = []

        # triangles
        self.tri_indices = []
        self.tri_poses = []
        self.tri_activations = []
        self.tri_materials = []
        self.tri_areas = []

        # edges (bending)
        self.edge_indices = []
        self.edge_rest_angle = []
        self.edge_rest_length = []
        self.edge_bending_properties = []

        # tetrahedra
        self.tet_indices = []
        self.tet_poses = []
        self.tet_activations = []
        self.tet_materials = []

        # muscles
        self.muscle_start = []
        self.muscle_params = []
        self.muscle_activations = []
        self.muscle_bodies = []
        self.muscle_points = []

        # rigid bodies
        self.body_mass = []
        self.body_inertia = []
        self.body_inv_mass = []
        self.body_inv_inertia = []
        self.body_com = []
        self.body_q = []
        self.body_qd = []
        self.body_key = []
        self.body_lock_inertia = []
        self.body_shapes = {-1: []}  # mapping from body to shapes
        self.body_world = []  # world index for each body
        self.body_color_groups: list[nparray] = []

        # rigid joints
        self.joint_parent = []  # index of the parent body                      (constant)
        self.joint_parents = {}  # mapping from child body to parent bodies
        self.joint_children = {}  # mapping from parent body to child bodies
        self.joint_child = []  # index of the child body                       (constant)
        self.joint_axis = []  # joint axis in joint parent anchor frame        (constant)
        self.joint_X_p = []  # frame of joint in parent                      (constant)
        self.joint_X_c = []  # frame of child com (in child coordinates)     (constant)
        self.joint_q = []
        self.joint_qd = []
        self.joint_cts = []
        self.joint_f = []

        self.joint_type = []
        self.joint_key = []
        self.joint_armature = []
        self.joint_act_mode = []  # Actuator mode per DOF (ActuatorMode.NONE=0, POSITION=1, VELOCITY=2, POSITION_VELOCITY=3, EFFORT=4)
        self.joint_target_ke = []
        self.joint_target_kd = []
        self.joint_limit_lower = []
        self.joint_limit_upper = []
        self.joint_limit_ke = []
        self.joint_limit_kd = []
        self.joint_target_pos = []
        self.joint_target_vel = []
        self.joint_effort_limit = []
        self.joint_velocity_limit = []
        self.joint_friction = []

        self.joint_twist_lower = []
        self.joint_twist_upper = []

        self.joint_enabled = []

        self.joint_q_start = []
        self.joint_qd_start = []
        self.joint_cts_start = []
        self.joint_dof_dim = []
        self.joint_world = []  # world index for each joint
        self.joint_articulation = []  # articulation index for each joint, -1 if not in any articulation

        self.articulation_start = []
        self.articulation_key = []
        self.articulation_world = []  # world index for each articulation

        self.joint_dof_count = 0
        self.joint_coord_count = 0
        self.joint_constraint_count = 0

        # current world index for entities being added to this builder.
        # set to -1 to create global entities shared across all worlds.
        self.current_world = -1

        self.up_axis: Axis = Axis.from_any(up_axis)
        self.gravity: float = gravity

        # Per-world gravity vectors, populated when worlds are created via begin_world()
        # Each entry is a tuple (gx, gy, gz) representing the gravity vector for that world
        self.world_gravity: list[tuple[float, float, float]] = []

        # contacts to be generated within the given distance margin to be generated at
        # every simulation substep (can be 0 if only one PBD solver iteration is used)
        self.rigid_contact_margin = 0.1

        # number of rigid contact points to allocate in the model during self.finalize() per world
        # if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
        self.num_rigid_contacts_per_world = None

        # equality constraints
        self.equality_constraint_type = []
        self.equality_constraint_body1 = []
        self.equality_constraint_body2 = []
        self.equality_constraint_anchor = []
        self.equality_constraint_relpose = []
        self.equality_constraint_torquescale = []
        self.equality_constraint_joint1 = []
        self.equality_constraint_joint2 = []
        self.equality_constraint_polycoef = []
        self.equality_constraint_key = []
        self.equality_constraint_enabled = []
        self.equality_constraint_world = []

        # mimic constraints
        self.constraint_mimic_joint0 = []
        self.constraint_mimic_joint1 = []
        self.constraint_mimic_coef0 = []
        self.constraint_mimic_coef1 = []
        self.constraint_mimic_enabled = []
        self.constraint_mimic_key = []
        self.constraint_mimic_world = []

        # per-world entity start indices
        self.particle_world_start = []
        self.body_world_start = []
        self.shape_world_start = []
        self.joint_world_start = []
        self.articulation_world_start = []
        self.equality_constraint_world_start = []
        self.joint_dof_world_start = []
        self.joint_coord_world_start = []
        self.joint_constraint_world_start = []

        # Custom attributes (user-defined per-frequency arrays)
        self.custom_attributes: dict[str, ModelBuilder.CustomAttribute] = {}
        # Incrementally maintained counts for custom string frequencies
        self._custom_frequency_counts: dict[str, int] = {}

    def add_shape_collision_filter_pair(self, shape_a: int, shape_b: int) -> None:
        """Add a collision filter pair in canonical order.

        Args:
            shape_a: First shape index
            shape_b: Second shape index
        """
        self.shape_collision_filter_pairs.append((min(shape_a, shape_b), max(shape_a, shape_b)))

    def add_custom_attribute(self, attribute: CustomAttribute) -> None:
        """
        Define a custom per-entity attribute to be added to the Model.
        See :ref:`custom_attributes` for more information.

        Args:
            attribute: The custom attribute to add.

        Example:

            .. doctest::

                builder = newton.ModelBuilder()
                builder.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name="my_attribute",
                        frequency=newton.Model.AttributeFrequency.BODY,
                        dtype=wp.float32,
                        default=20.0,
                        assignment=newton.Model.AttributeAssignment.MODEL,
                        namespace="my_namespace",
                    )
                )
                builder.add_body(custom_attributes={"my_namespace:my_attribute": 30.0})
                builder.add_body()  # we leave out the custom_attributes, so the attribute will use the default value 20.0
                model = builder.finalize()
                # the model has now a Model.AttributeNamespace object with the name "my_namespace"
                # and an attribute "my_attribute" that is a wp.array of shape (body_count, 1)
                # with the default value 20.0
                assert np.allclose(model.my_namespace.my_attribute.numpy(), [30.0, 20.0])
        """
        key = attribute.key

        existing = self.custom_attributes.get(key)
        if existing:
            # validate that specification matches exactly
            if (
                existing.frequency != attribute.frequency
                or existing.dtype != attribute.dtype
                or existing.assignment != attribute.assignment
                or existing.namespace != attribute.namespace
                or existing.references != attribute.references
            ):
                raise ValueError(f"Custom attribute '{key}' already exists with incompatible spec")
            return

        self.custom_attributes[key] = attribute
        # Initialize frequency count for string frequencies
        if attribute.is_custom_frequency:
            freq_key = attribute.frequency_key
            if freq_key not in self._custom_frequency_counts:
                self._custom_frequency_counts[freq_key] = 0

    def has_custom_attribute(self, key: str) -> bool:
        """Check if a custom attribute is defined."""
        return key in self.custom_attributes

    def get_custom_attributes_by_frequency(
        self, frequencies: Sequence[Model.AttributeFrequency]
    ) -> list[CustomAttribute]:
        """
        Get custom attributes by frequency.
        This is useful for processing custom attributes for different kinds of simulation objects.
        For example, you can get all the custom attributes for bodies, shapes, joints, etc.

        Args:
            frequencies: The frequencies to get custom attributes for.

        Returns:
            A list of custom attributes.
        """
        return [attr for attr in self.custom_attributes.values() if attr.frequency in frequencies]

    def get_custom_frequency_keys(self) -> set[str]:
        """Return set of custom frequency keys (string frequencies) defined in this builder."""
        return set(self._custom_frequency_counts.keys())

    def add_custom_values(self, **kwargs: Any) -> dict[str, int]:
        """Append values to custom attributes with custom string frequencies.

        Each keyword argument specifies an attribute key and the value to append. Values are
        stored in a list and appended sequentially for robust indexing. Only works with
        attributes that have a custom string frequency (not built-in enum frequencies).

        This is useful for custom entity types that aren't built into the model,
        such as user-defined groupings or solver-specific data.

        Args:
            **kwargs: Mapping of attribute keys to values. Keys should be the full
                attribute key (e.g., ``"mujoco:pair_geom1"`` or just ``"my_attr"`` if no namespace).

        Returns:
            Dict mapping attribute keys to the index where each value was added.
            If all attributes had the same count before the call, all indices will be equal.

        Raises:
            AttributeError: If an attribute key is not defined.
            TypeError: If an attribute has an enum frequency (must have custom frequency).

        Example:
            .. code-block:: python

                builder.add_custom_values(
                    **{
                        "mujoco:pair_geom1": 0,
                        "mujoco:pair_geom2": 1,
                        "mujoco:pair_world": builder.current_world,
                    }
                )
                # Returns: {'mujoco:pair_geom1': 0, 'mujoco:pair_geom2': 0, 'mujoco:pair_world': 0}
        """
        indices: dict[str, int] = {}
        frequency_indices: dict[str, int] = {}  # Track indices assigned per frequency in this call

        for key, value in kwargs.items():
            attr = self.custom_attributes.get(key)
            if attr is None:
                raise AttributeError(
                    f"Custom attribute '{key}' is not defined. Please declare it first using add_custom_attribute()."
                )
            if not attr.is_custom_frequency:
                raise TypeError(
                    f"Custom attribute '{key}' has frequency={attr.frequency}, "
                    f"but add_custom_values() only works with custom frequency attributes."
                )

            # Ensure attr.values is initialized
            if attr.values is None:
                attr.values = []

            freq_key = attr.frequency_key

            # Determine index for this frequency (same index for all attrs with same frequency in this call)
            if freq_key not in frequency_indices:
                # First attribute with this frequency - use authoritative counter
                current_count = self._custom_frequency_counts.get(freq_key, 0)
                frequency_indices[freq_key] = current_count

                # Update authoritative counter for this frequency
                self._custom_frequency_counts[freq_key] = current_count + 1

            idx = frequency_indices[freq_key]

            # Ensure attr.values has length at least idx+1, padding with None as needed
            while len(attr.values) <= idx:
                attr.values.append(None)

            # Assign value at the correct index
            attr.values[idx] = value
            indices[key] = idx
        return indices

    def _process_custom_attributes(
        self,
        entity_index: int | list[int],
        custom_attrs: dict[str, Any],
        expected_frequency: Model.AttributeFrequency,
    ) -> None:
        """Process custom attributes from kwargs and assign them to an entity.

        This method validates that custom attributes exist with the correct frequency,
        then assigns values to the specific entity. The assignment is inferred from the
        attribute definition.

        Attribute names can optionally include a namespace prefix in the format "namespace:attr_name".
        If no namespace prefix is provided, the attribute is assumed to be in the default namespace (None).

        Args:
            entity_index: Index of the entity (body, shape, joint, etc.). Can be a single index or a list of indices.
            custom_attrs: Dictionary of custom attribute names to values.
                Keys can be "attr_name" or "namespace:attr_name". Values can be a single value or a list of values.
            expected_frequency: Expected frequency for these attributes.
        """
        for attr_key, value in custom_attrs.items():
            # Parse namespace prefix if present (format: "namespace:attr_name" or "attr_name")
            full_key = attr_key

            # Ensure the custom attribute is defined
            custom_attr = self.custom_attributes.get(full_key)
            if custom_attr is None:
                raise AttributeError(
                    f"Custom attribute '{full_key}' is not defined. "
                    f"Please declare it first using add_custom_attribute()."
                )

            # Validate frequency matches
            if custom_attr.frequency != expected_frequency:
                raise ValueError(
                    f"Custom attribute '{full_key}' has frequency {custom_attr.frequency}, "
                    f"but expected {expected_frequency} for this entity type"
                )

            # Set the value for this specific entity
            if custom_attr.values is None:
                custom_attr.values = {}

            # Fill in the value(s)
            if isinstance(entity_index, list):
                value_is_sequence = isinstance(value, (list, tuple))
                if isinstance(value, np.ndarray):
                    value_is_sequence = value.ndim != 0
                if value_is_sequence:
                    if len(value) != len(entity_index):
                        raise ValueError(f"Expected {len(entity_index)} values, got {len(value)}")
                    custom_attr.values.update(zip(entity_index, value, strict=False))
                else:
                    custom_attr.values.update((idx, value) for idx in entity_index)
            else:
                custom_attr.values[entity_index] = value

    def _process_joint_custom_attributes(
        self,
        joint_index: int,
        custom_attrs: dict[str, Any],
    ) -> None:
        """Process custom attributes from kwargs for joints, supporting multiple frequencies.

        Joint attributes are processed based on their declared frequency:
        - JOINT frequency: Single value per joint
        - JOINT_DOF frequency: List or dict of values for each DOF
        - JOINT_COORD frequency: List or dict of values for each coordinate

        For DOF and COORD attributes, values can be:
        - A list with length matching the joint's DOF/coordinate count (all DOFs get values)
        - A dict mapping DOF/coord indices to values (only specified indices get values, rest use defaults)
        - For single-DOF joints with JOINT_DOF frequency: a single Warp vector/matrix value

        When using dict format, unspecified indices will be filled with the attribute's default value during finalization.

        Args:
            joint_index: Index of the joint
            custom_attrs: Dictionary of custom attribute names to values
        """

        def apply_indexed_values(
            *,
            value: Any,
            attr_key: str,
            expected_frequency: Model.AttributeFrequency,
            index_start: int,
            index_count: int,
            index_label: str,
            count_label: str,
            length_error_template: str,
        ) -> None:
            if isinstance(value, dict):
                for offset, offset_value in value.items():
                    if not isinstance(offset, int):
                        raise TypeError(
                            f"{expected_frequency.name} attribute '{attr_key}' dict keys must be integers "
                            f"({index_label} indices), got {type(offset)}"
                        )
                    if offset < 0 or offset >= index_count:
                        raise ValueError(
                            f"{expected_frequency.name} attribute '{attr_key}' has invalid {index_label} index "
                            f"{offset} (joint has {index_count} {count_label})"
                        )
                    self._process_custom_attributes(
                        entity_index=index_start + offset,
                        custom_attrs={attr_key: offset_value},
                        expected_frequency=expected_frequency,
                    )
                return

            value_sanitized = value
            if not isinstance(value_sanitized, (list, tuple)) and index_count == 1:
                value_sanitized = [value_sanitized]

            actual = len(value_sanitized)
            if actual != index_count:
                raise ValueError(length_error_template.format(attr_key=attr_key, actual=actual, expected=index_count))

            for i, indexed_value in enumerate(value_sanitized):
                self._process_custom_attributes(
                    entity_index=index_start + i,
                    custom_attrs={attr_key: indexed_value},
                    expected_frequency=expected_frequency,
                )

        for attr_key, value in custom_attrs.items():
            # Look up the attribute to determine its frequency
            custom_attr = self.custom_attributes.get(attr_key)
            if custom_attr is None:
                raise AttributeError(
                    f"Custom attribute '{attr_key}' is not defined. "
                    f"Please declare it first using add_custom_attribute()."
                )

            # Process based on declared frequency
            if custom_attr.frequency == Model.AttributeFrequency.JOINT:
                # Single value per joint
                self._process_custom_attributes(
                    entity_index=joint_index,
                    custom_attrs={attr_key: value},
                    expected_frequency=Model.AttributeFrequency.JOINT,
                )

            elif custom_attr.frequency in (
                Model.AttributeFrequency.JOINT_DOF,
                Model.AttributeFrequency.JOINT_COORD,
                Model.AttributeFrequency.JOINT_CONSTRAINT,
            ):
                freq = custom_attr.frequency
                freq_config = {
                    Model.AttributeFrequency.JOINT_DOF: (
                        self.joint_qd_start,
                        self.joint_dof_count,
                        "DOF",
                        "DOFs",
                    ),
                    Model.AttributeFrequency.JOINT_COORD: (
                        self.joint_q_start,
                        self.joint_coord_count,
                        "coord",
                        "coordinates",
                    ),
                    Model.AttributeFrequency.JOINT_CONSTRAINT: (
                        self.joint_cts_start,
                        self.joint_constraint_count,
                        "constraint",
                        "constraints",
                    ),
                }
                start_array, total_count, index_label, count_label = freq_config[freq]

                index_start = start_array[joint_index]
                if joint_index + 1 < len(start_array):
                    index_end = start_array[joint_index + 1]
                else:
                    index_end = total_count

                apply_indexed_values(
                    value=value,
                    attr_key=attr_key,
                    expected_frequency=freq,
                    index_start=index_start,
                    index_count=index_end - index_start,
                    index_label=index_label,
                    count_label=count_label,
                    length_error_template=(
                        f"{freq.name} attribute '{{attr_key}}' has {{actual}} values "
                        f"but joint has {{expected}} {count_label}"
                    ),
                )

            else:
                raise ValueError(
                    f"Custom attribute '{attr_key}' has unsupported frequency {custom_attr.frequency} for joints"
                )

    @property
    def default_site_cfg(self) -> ShapeConfig:
        """Returns a ShapeConfig configured for sites (non-colliding reference points).

        This config has all site invariants enforced:
        - is_site = True
        - has_shape_collision = False
        - has_particle_collision = False
        - density = 0.0
        - collision_group = 0

        Returns:
            ShapeConfig: A new configuration suitable for creating sites.
        """
        cfg = self.ShapeConfig()
        cfg.mark_as_site()
        return cfg

    @property
    def up_vector(self) -> Vec3:
        """
        Returns the 3D unit vector corresponding to the current up axis (read-only).

        This property computes the up direction as a 3D vector based on the value of :attr:`up_axis`.
        For example, if ``up_axis`` is ``Axis.Z``, this returns ``(0, 0, 1)``.

        Returns:
            Vec3: The 3D up vector corresponding to the current up axis.
        """
        return axis_to_vec3(self.up_axis)

    @up_vector.setter
    def up_vector(self, _):
        raise AttributeError(
            "The 'up_vector' property is read-only and cannot be set. Instead, use 'up_axis' to set the up axis."
        )

    # region counts
    @property
    def shape_count(self):
        """
        The number of shapes in the model.
        """
        return len(self.shape_type)

    @property
    def body_count(self):
        """
        The number of rigid bodies in the model.
        """
        return len(self.body_q)

    @property
    def joint_count(self):
        """
        The number of joints in the model.
        """
        return len(self.joint_type)

    @property
    def particle_count(self):
        """
        The number of particles in the model.
        """
        return len(self.particle_q)

    @property
    def tri_count(self):
        """
        The number of triangles in the model.
        """
        return len(self.tri_poses)

    @property
    def tet_count(self):
        """
        The number of tetrahedra in the model.
        """
        return len(self.tet_poses)

    @property
    def edge_count(self):
        """
        The number of edges (for bending) in the model.
        """
        return len(self.edge_rest_angle)

    @property
    def spring_count(self):
        """
        The number of springs in the model.
        """
        return len(self.spring_rest_length)

    @property
    def muscle_count(self):
        """
        The number of muscles in the model.
        """
        return len(self.muscle_start)

    @property
    def articulation_count(self):
        """
        The number of articulations in the model.
        """
        return len(self.articulation_start)

    # endregion

    def replicate(
        self,
        builder: ModelBuilder,
        num_worlds: int,
        spacing: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Replicates the given builder multiple times, offsetting each copy according to the supplied spacing.

        This method is useful for creating multiple instances of a sub-model (e.g., robots, scenes)
        arranged in a regular grid or along a line. Each copy is offset in space by a multiple of the
        specified spacing vector, and all entities from each copy are assigned to a new world.

        Note:
            For visual separation of worlds, it is recommended to use the viewer's
            `set_world_offsets()` method instead of physical spacing. This improves numerical
            stability by keeping all worlds at the origin in the physics simulation.

        Args:
            builder (ModelBuilder): The builder to replicate. All entities from this builder will be copied.
            num_worlds (int): The number of worlds to create.
            spacing (tuple[float, float, float], optional): The spacing between each copy along each axis.
                For example, (5.0, 5.0, 0.0) arranges copies in a 2D grid in the XY plane.
                Defaults to (0.0, 0.0, 0.0).
        """
        offsets = compute_world_offsets(num_worlds, spacing, self.up_axis)
        xform = wp.transform_identity()
        for i in range(num_worlds):
            xform[:3] = offsets[i]
            self.add_world(builder, xform=xform)

    def add_articulation(
        self, joints: list[int], key: str | None = None, custom_attributes: dict[str, Any] | None = None
    ):
        """
        Adds an articulation to the model from a list of joint indices.

        The articulation is a set of joints that must be contiguous and monotonically increasing.
        Some functions, such as forward kinematics :func:`newton.eval_fk`, are parallelized over articulations.

        Args:
            joints: List of joint indices to include in the articulation. Must be contiguous and monotonic.
            key: The key of the articulation. If None, a default key will be created.
            custom_attributes: Dictionary of custom attribute values for ARTICULATION frequency attributes.

        Raises:
            ValueError: If joints are not contiguous, not monotonic, or belong to different worlds.

        Example:
            .. code-block:: python

                link1 = builder.add_link(...)
                link2 = builder.add_link(...)
                link3 = builder.add_link(...)

                joint1 = builder.add_joint_revolute(parent=-1, child=link1)
                joint2 = builder.add_joint_revolute(parent=link1, child=link2)
                joint3 = builder.add_joint_revolute(parent=link2, child=link3)

                # Create articulation from the joints
                builder.add_articulation([joint1, joint2, joint3])
        """
        if not joints:
            raise ValueError("Cannot create an articulation with no joints")

        # Sort joints to ensure we can validate them properly
        sorted_joints = sorted(joints)

        # Validate joints are monotonically increasing (no duplicates)
        if sorted_joints != joints:
            raise ValueError(
                f"Joints must be provided in monotonically increasing order. Got {joints}, expected {sorted_joints}"
            )

        # Validate joints are contiguous
        for i in range(1, len(sorted_joints)):
            if sorted_joints[i] != sorted_joints[i - 1] + 1:
                raise ValueError(
                    f"Joints must be contiguous. Got indices {sorted_joints}, but there is a gap between "
                    f"{sorted_joints[i - 1]} and {sorted_joints[i]}. Create all joints for an articulation "
                    f"before creating joints for another articulation."
                )

        # Validate all joints exist and don't already belong to an articulation
        for joint_idx in joints:
            if joint_idx < 0 or joint_idx >= len(self.joint_type):
                raise ValueError(
                    f"Joint index {joint_idx} is out of range. Valid range is 0 to {len(self.joint_type) - 1}"
                )
            if self.joint_articulation[joint_idx] >= 0:
                existing_art = self.joint_articulation[joint_idx]
                raise ValueError(
                    f"Joint {joint_idx} ('{self.joint_key[joint_idx]}') already belongs to articulation {existing_art} "
                    f"('{self.articulation_key[existing_art]}'). Each joint can only belong to one articulation."
                )

        # Validate all joints belong to the same world (current world)
        for joint_idx in joints:
            if joint_idx < len(self.joint_world) and self.joint_world[joint_idx] != self.current_world:
                raise ValueError(
                    f"Joint {joint_idx} belongs to world {self.joint_world[joint_idx]}, but current world is "
                    f"{self.current_world}. All joints in an articulation must belong to the same world."
                )

        # Basic tree structure validation (check for cycles, single parent)
        # Build a simple tree structure check - each child should have only one parent in this articulation
        child_to_parent = {}
        for joint_idx in joints:
            child = self.joint_child[joint_idx]
            parent = self.joint_parent[joint_idx]
            if child in child_to_parent and child_to_parent[child] != parent:
                raise ValueError(
                    f"Body {child} has multiple parents in this articulation: {child_to_parent[child]} and {parent}. "
                    f"This creates an invalid tree structure. Loop-closing joints must not be part of an articulation."
                )
            child_to_parent[child] = parent

        # Store the articulation using the first joint's index as the start
        articulation_idx = self.articulation_count
        self.articulation_start.append(sorted_joints[0])
        self.articulation_key.append(key or f"articulation_{articulation_idx}")
        self.articulation_world.append(self.current_world)

        # Mark all joints as belonging to this articulation
        for joint_idx in joints:
            self.joint_articulation[joint_idx] = articulation_idx

        # Process custom attributes for this articulation
        if custom_attributes:
            self._process_custom_attributes(
                entity_index=articulation_idx,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.ARTICULATION,
            )

    # region importers
    def add_urdf(
        self,
        source: str,
        *,
        xform: Transform | None = None,
        floating: bool | None = None,
        base_joint: dict | None = None,
        parent_body: int = -1,
        scale: float = 1.0,
        hide_visuals: bool = False,
        parse_visuals_as_colliders: bool = False,
        up_axis: AxisType = Axis.Z,
        force_show_colliders: bool = False,
        enable_self_collisions: bool = True,
        ignore_inertial_definitions: bool = True,
        ensure_nonstatic_links: bool = True,
        static_link_mass: float = 1e-2,
        joint_ordering: Literal["bfs", "dfs"] | None = "dfs",
        bodies_follow_joint_ordering: bool = True,
        collapse_fixed_joints: bool = False,
        mesh_maxhullvert: int | None = None,
        force_position_velocity_actuation: bool = False,
    ):
        """
        Parses a URDF file and adds the bodies and joints to the given ModelBuilder.

        Args:
            source (str): The filename of the URDF file to parse, or the URDF XML string content.
            xform (Transform): The transform to apply to the root body. If None, the transform is set to identity.
            floating (bool or None): Controls the base joint type for the root body.

                - ``None`` (default): Uses format-specific default (creates a FIXED joint for URDF).
                - ``True``: Creates a FREE joint with 6 DOF (3 translation + 3 rotation). Only valid when
                  ``parent_body == -1`` since FREE joints must connect to world frame.
                - ``False``: Creates a FIXED joint (0 DOF).

                Cannot be specified together with ``base_joint``.
            base_joint (dict): Custom joint specification for connecting the root body to the world
                (or to ``parent_body`` if specified). This parameter enables hierarchical composition with
                custom mobility. Dictionary with joint parameters as accepted by
                :meth:`ModelBuilder.add_joint` (e.g., joint type, axes, limits, stiffness).

                Cannot be specified together with ``floating``.
            parent_body (int): Parent body index for hierarchical composition. If specified, attaches the
                imported root body to this existing body, making them part of the same kinematic articulation.
                The connection type is determined by ``floating`` or ``base_joint``. If ``-1`` (default),
                the root connects to the world frame. **Restriction**: Only the most recently added
                articulation can be used as parent; attempting to attach to an older articulation will raise
                a ``ValueError``.

                .. note::
                   Valid combinations of ``floating``, ``base_joint``, and ``parent_body``:

                   .. list-table::
                      :header-rows: 1
                      :widths: 15 15 15 55

                      * - floating
                        - base_joint
                        - parent_body
                        - Result
                      * - ``None``
                        - ``None``
                        - ``-1``
                        - Format default (URDF: FIXED joint)
                      * - ``True``
                        - ``None``
                        - ``-1``
                        - FREE joint to world (6 DOF)
                      * - ``False``
                        - ``None``
                        - ``-1``
                        - FIXED joint to world (0 DOF)
                      * - ``None``
                        - ``{dict}``
                        - ``-1``
                        - Custom joint to world (e.g., D6)
                      * - ``False``
                        - ``None``
                        - ``body_idx``
                        - FIXED joint to parent body
                      * - ``None``
                        - ``{dict}``
                        - ``body_idx``
                        - Custom joint to parent body (e.g., D6)
                      * - *explicitly set*
                        - *explicitly set*
                        - *any*
                        - ❌ Error: mutually exclusive (cannot specify both)
                      * - ``True``
                        - ``None``
                        - ``body_idx``
                        - ❌ Error: FREE joints require world frame

            scale (float): The scaling factor to apply to the imported mechanism.
            hide_visuals (bool): If True, hide visual shapes.
            parse_visuals_as_colliders (bool): If True, the geometry defined under the `<visual>` tags is used for collision handling instead of the `<collision>` geometries.
            up_axis (AxisType): The up axis of the URDF. This is used to transform the URDF to the builder's up axis. It also determines the up axis of capsules and cylinders in the URDF. The default is Z.
            force_show_colliders (bool): If True, the collision shapes are always shown, even if there are visual shapes.
            enable_self_collisions (bool): If True, self-collisions are enabled.
            ignore_inertial_definitions (bool): If True, the inertial parameters defined in the URDF are ignored and the inertia is calculated from the shape geometry.
            ensure_nonstatic_links (bool): If True, links with zero mass are given a small mass (see `static_link_mass`) to ensure they are dynamic.
            static_link_mass (float): The mass to assign to links with zero mass (if `ensure_nonstatic_links` is set to True).
            joint_ordering (str): The ordering of the joints in the simulation. Can be either "bfs" or "dfs" for breadth-first or depth-first search, or ``None`` to keep joints in the order in which they appear in the URDF. Default is "dfs".
            bodies_follow_joint_ordering (bool): If True, the bodies are added to the builder in the same order as the joints (parent then child body). Otherwise, bodies are added in the order they appear in the URDF. Default is True.
            collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.
            mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.
            force_position_velocity_actuation (bool): If True and both position (stiffness) and velocity
                (damping) gains are non-zero, joints use :attr:`~newton.ActuatorMode.POSITION_VELOCITY` actuation mode.
                If False (default), actuator modes are inferred per joint via :func:`newton.ActuatorMode.from_gains`:
                :attr:`~newton.ActuatorMode.POSITION` if stiffness > 0, :attr:`~newton.ActuatorMode.VELOCITY` if only
                damping > 0, :attr:`~newton.ActuatorMode.EFFORT` if a drive is present but both gains are zero
                (direct torque control), or :attr:`~newton.ActuatorMode.NONE` if no drive/actuation is applied.
        """
        from ..utils.import_urdf import parse_urdf  # noqa: PLC0415

        return parse_urdf(
            self,
            source,
            xform=xform,
            floating=floating,
            base_joint=base_joint,
            parent_body=parent_body,
            scale=scale,
            hide_visuals=hide_visuals,
            parse_visuals_as_colliders=parse_visuals_as_colliders,
            up_axis=up_axis,
            force_show_colliders=force_show_colliders,
            enable_self_collisions=enable_self_collisions,
            ignore_inertial_definitions=ignore_inertial_definitions,
            ensure_nonstatic_links=ensure_nonstatic_links,
            static_link_mass=static_link_mass,
            joint_ordering=joint_ordering,
            bodies_follow_joint_ordering=bodies_follow_joint_ordering,
            collapse_fixed_joints=collapse_fixed_joints,
            mesh_maxhullvert=mesh_maxhullvert,
            force_position_velocity_actuation=force_position_velocity_actuation,
        )

    def add_usd(
        self,
        source,
        *,
        xform: Transform | None = None,
        floating: bool | None = None,
        base_joint: dict | None = None,
        parent_body: int = -1,
        only_load_enabled_rigid_bodies: bool = False,
        only_load_enabled_joints: bool = True,
        joint_drive_gains_scaling: float = 1.0,
        verbose: bool = False,
        ignore_paths: list[str] | None = None,
        collapse_fixed_joints: bool = False,
        enable_self_collisions: bool = True,
        apply_up_axis_from_stage: bool = False,
        root_path: str = "/",
        joint_ordering: Literal["bfs", "dfs"] | None = "dfs",
        bodies_follow_joint_ordering: bool = True,
        skip_mesh_approximation: bool = False,
        load_sites: bool = True,
        load_visual_shapes: bool = True,
        hide_collision_shapes: bool = False,
        parse_mujoco_options: bool = True,
        mesh_maxhullvert: int | None = None,
        schema_resolvers: list[SchemaResolver] | None = None,
        force_position_velocity_actuation: bool = False,
    ) -> dict[str, Any]:
        """Parses a Universal Scene Description (USD) stage containing UsdPhysics schema definitions for rigid-body articulations and adds the bodies, shapes and joints to the given ModelBuilder.

        The USD description has to be either a path (file name or URL), or an existing USD stage instance that implements the `Stage <https://openusd.org/dev/api/class_usd_stage.html>`_ interface.

        See :ref:`usd_parsing` for more information.

        Args:
            source (str | pxr.Usd.Stage): The file path to the USD file, or an existing USD stage instance.
            xform (Transform): The transform to apply to the entire scene.
            floating (bool or None): Controls the base joint type for the root body (bodies not connected as
                a child to any joint).

                - ``None`` (default): Uses format-specific default (creates a FREE joint for USD bodies without joints).
                - ``True``: Creates a FREE joint with 6 DOF (3 translation + 3 rotation). Only valid when
                  ``parent_body == -1`` since FREE joints must connect to world frame.
                - ``False``: Creates a FIXED joint (0 DOF).

                Cannot be specified together with ``base_joint``.
            base_joint (dict): Custom joint specification for connecting the root body to the world
                (or to ``parent_body`` if specified). This parameter enables hierarchical composition with
                custom mobility. Dictionary with joint parameters as accepted by
                :meth:`ModelBuilder.add_joint` (e.g., joint type, axes, limits, stiffness).

                Cannot be specified together with ``floating``.
            parent_body (int): Parent body index for hierarchical composition. If specified, attaches the
                imported root body to this existing body, making them part of the same kinematic articulation.
                The connection type is determined by ``floating`` or ``base_joint``. If ``-1`` (default),
                the root connects to the world frame. **Restriction**: Only the most recently added
                articulation can be used as parent; attempting to attach to an older articulation will raise
                a ``ValueError``.

                .. note::
                   Valid combinations of ``floating``, ``base_joint``, and ``parent_body``:

                   .. list-table::
                      :header-rows: 1
                      :widths: 15 15 15 55

                      * - floating
                        - base_joint
                        - parent_body
                        - Result
                      * - ``None``
                        - ``None``
                        - ``-1``
                        - Format default (USD: FREE joint for bodies without joints)
                      * - ``True``
                        - ``None``
                        - ``-1``
                        - FREE joint to world (6 DOF)
                      * - ``False``
                        - ``None``
                        - ``-1``
                        - FIXED joint to world (0 DOF)
                      * - ``None``
                        - ``{dict}``
                        - ``-1``
                        - Custom joint to world (e.g., D6)
                      * - ``False``
                        - ``None``
                        - ``body_idx``
                        - FIXED joint to parent body
                      * - ``None``
                        - ``{dict}``
                        - ``body_idx``
                        - Custom joint to parent body (e.g., D6)
                      * - *explicitly set*
                        - *explicitly set*
                        - *any*
                        - ❌ Error: mutually exclusive (cannot specify both)
                      * - ``True``
                        - ``None``
                        - ``body_idx``
                        - ❌ Error: FREE joints require world frame

            only_load_enabled_rigid_bodies (bool): If True, only rigid bodies which do not have `physics:rigidBodyEnabled` set to False are loaded.
            only_load_enabled_joints (bool): If True, only joints which do not have `physics:jointEnabled` set to False are loaded.
            joint_drive_gains_scaling (float): The default scaling of the PD control gains (stiffness and damping), if not set in the PhysicsScene with as "newton:joint_drive_gains_scaling".
            verbose (bool): If True, print additional information about the parsed USD file. Default is False.
            ignore_paths (List[str]): A list of regular expressions matching prim paths to ignore.
            collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged. Only considered if not set on the PhysicsScene as "newton:collapse_fixed_joints".
            enable_self_collisions (bool): Determines the default behavior of whether self-collisions are enabled for all shapes within an articulation. If an articulation has the attribute ``physxArticulation:enabledSelfCollisions`` defined, this attribute takes precedence.
            apply_up_axis_from_stage (bool): If True, the up axis of the stage will be used to set :attr:`newton.ModelBuilder.up_axis`. Otherwise, the stage will be rotated such that its up axis aligns with the builder's up axis. Default is False.
            root_path (str): The USD path to import, defaults to "/".
            joint_ordering (str): The ordering of the joints in the simulation. Can be either "bfs" or "dfs" for breadth-first or depth-first search, or ``None`` to keep joints in the order in which they appear in the USD. Default is "dfs".
            bodies_follow_joint_ordering (bool): If True, the bodies are added to the builder in the same order as the joints (parent then child body). Otherwise, bodies are added in the order they appear in the USD. Default is True.
            skip_mesh_approximation (bool): If True, mesh approximation is skipped. Otherwise, meshes are approximated according to the ``physics:approximation`` attribute defined on the UsdPhysicsMeshCollisionAPI (if it is defined). Default is False.
            load_sites (bool): If True, sites (prims with MjcSiteAPI) are loaded as non-colliding reference points. If False, sites are ignored. Default is True.
            load_visual_shapes (bool): If True, non-physics visual geometry is loaded. If False, visual-only shapes are ignored (sites are still controlled by ``load_sites``). Default is True.
            hide_collision_shapes (bool): If True, collision shapes are hidden. Default is False.
            parse_mujoco_options (bool): Whether MuJoCo solver options from the PhysicsScene should be parsed. If False, solver options are not loaded and custom attributes retain their default values. Default is True.
            mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes. Note that an authored ``newton:maxHullVertices`` attribute on any shape with a ``NewtonMeshCollisionAPI`` will take priority over this value.
            schema_resolvers (list[SchemaResolver]): Resolver instances in priority order. Default is to only parse Newton-specific attributes.
                Schema resolvers collect per-prim "solver-specific" attributes, see :ref:`schema_resolvers` for more information.
                These include namespaced attributes such as ``newton:*``, ``physx*``
                (e.g., ``physxScene:*``, ``physxRigidBody:*``, ``physxSDFMeshCollision:*``), and ``mjc:*`` that
                are authored in the USD but not strictly required to build the simulation. This is useful for
                inspection, experimentation, or custom pipelines that read these values via
                :attr:`newton.usd.SchemaResolverManager.schema_attrs`.

                .. note::
                    Using the ``schema_resolvers`` argument is an experimental feature that may be removed or changed significantly in the future.
            force_position_velocity_actuation (bool): If True and both stiffness (kp) and damping (kd)
                are non-zero, joints use :attr:`~newton.ActuatorMode.POSITION_VELOCITY` actuation mode.
                If False (default), actuator modes are inferred per joint via :func:`newton.ActuatorMode.from_gains`:
                :attr:`~newton.ActuatorMode.POSITION` if stiffness > 0, :attr:`~newton.ActuatorMode.VELOCITY` if only
                damping > 0, :attr:`~newton.ActuatorMode.EFFORT` if a drive is present but both gains are zero
                (direct torque control), or :attr:`~newton.ActuatorMode.NONE` if no drive/actuation is applied.

        Returns:
            dict: Dictionary with the following entries:

            .. list-table::
                :widths: 25 75

                * - ``"fps"``
                  - USD stage frames per second
                * - ``"duration"``
                  - Difference between end time code and start time code of the USD stage
                * - ``"up_axis"``
                  - :class:`Axis` representing the stage's up axis ("X", "Y", or "Z")
                * - ``"path_body_map"``
                  - Mapping from prim path (str) of a rigid body prim (e.g. that implements the PhysicsRigidBodyAPI) to the respective body index in :class:`~newton.ModelBuilder`
                * - ``"path_joint_map"``
                  - Mapping from prim path (str) of a joint prim (e.g. that implements the PhysicsJointAPI) to the respective joint index in :class:`~newton.ModelBuilder`
                * - ``"path_shape_map"``
                  - Mapping from prim path (str) of the UsdGeom to the respective shape index in :class:`~newton.ModelBuilder`
                * - ``"path_shape_scale"``
                  - Mapping from prim path (str) of the UsdGeom to its respective 3D world scale
                * - ``"mass_unit"``
                  - The stage's Kilograms Per Unit (KGPU) definition (1.0 by default)
                * - ``"linear_unit"``
                  - The stage's Meters Per Unit (MPU) definition (1.0 by default)
                * - ``"scene_attributes"``
                  - Dictionary of all attributes applied to the PhysicsScene prim
                * - ``"collapse_results"``
                  - Dictionary returned by :meth:`newton.ModelBuilder.collapse_fixed_joints` if ``collapse_fixed_joints`` is True, otherwise None.
                * - ``"physics_dt"``
                  - The resolved physics scene time step (float or None)
                * - ``"schema_attrs"``
                  - Dictionary of collected per-prim schema attributes (dict)
                * - ``"max_solver_iterations"``
                  - The resolved maximum solver iterations (int or None)
                * - ``"path_body_relative_transform"``
                  - Mapping from prim path to relative transform for bodies merged via ``collapse_fixed_joints``
                * - ``"path_original_body_map"``
                  - Mapping from prim path to original body index before ``collapse_fixed_joints``
        """
        from ..utils.import_usd import parse_usd  # noqa: PLC0415

        return parse_usd(
            self,
            source,
            xform=xform,
            floating=floating,
            base_joint=base_joint,
            parent_body=parent_body,
            only_load_enabled_rigid_bodies=only_load_enabled_rigid_bodies,
            only_load_enabled_joints=only_load_enabled_joints,
            joint_drive_gains_scaling=joint_drive_gains_scaling,
            verbose=verbose,
            ignore_paths=ignore_paths,
            collapse_fixed_joints=collapse_fixed_joints,
            enable_self_collisions=enable_self_collisions,
            apply_up_axis_from_stage=apply_up_axis_from_stage,
            root_path=root_path,
            joint_ordering=joint_ordering,
            bodies_follow_joint_ordering=bodies_follow_joint_ordering,
            skip_mesh_approximation=skip_mesh_approximation,
            load_sites=load_sites,
            load_visual_shapes=load_visual_shapes,
            hide_collision_shapes=hide_collision_shapes,
            parse_mujoco_options=parse_mujoco_options,
            mesh_maxhullvert=mesh_maxhullvert,
            schema_resolvers=schema_resolvers,
            force_position_velocity_actuation=force_position_velocity_actuation,
        )

    def add_mjcf(
        self,
        source: str,
        *,
        xform: Transform | None = None,
        floating: bool | None = None,
        base_joint: dict | None = None,
        parent_body: int = -1,
        armature_scale: float = 1.0,
        scale: float = 1.0,
        hide_visuals: bool = False,
        parse_visuals_as_colliders: bool = False,
        parse_meshes: bool = True,
        parse_sites: bool = True,
        parse_visuals: bool = True,
        parse_mujoco_options: bool = True,
        up_axis: AxisType = Axis.Z,
        ignore_names: Sequence[str] = (),
        ignore_classes: Sequence[str] = (),
        visual_classes: Sequence[str] = ("visual",),
        collider_classes: Sequence[str] = ("collision",),
        no_class_as_colliders: bool = True,
        force_show_colliders: bool = False,
        enable_self_collisions: bool = True,
        ignore_inertial_definitions: bool = True,
        ensure_nonstatic_links: bool = True,
        static_link_mass: float = 1e-2,
        collapse_fixed_joints: bool = False,
        verbose: bool = False,
        skip_equality_constraints: bool = False,
        convert_3d_hinge_to_ball_joints: bool = False,
        mesh_maxhullvert: int | None = None,
        ctrl_direct: bool = False,
        path_resolver: Callable[[str | None, str], str] | None = None,
    ):
        """
        Parses MuJoCo XML (MJCF) file and adds the bodies and joints to the given ModelBuilder.
        MuJoCo-specific custom attributes are registered on the builder automatically.

        Args:
            source (str): The filename of the MuJoCo file to parse, or the MJCF XML string content.
            xform (Transform): The transform to apply to the imported mechanism.
            floating (bool or None): Controls the base joint type for the root body.

                - ``None`` (default): Uses format-specific default (honors ``<freejoint>`` tags in MJCF,
                  otherwise creates a FIXED joint).
                - ``True``: Creates a FREE joint with 6 DOF (3 translation + 3 rotation). Only valid when
                  ``parent_body == -1`` since FREE joints must connect to world frame.
                - ``False``: Creates a FIXED joint (0 DOF).

                Cannot be specified together with ``base_joint``.
            base_joint (dict): Custom joint specification for connecting the root body to the world
                (or to ``parent_body`` if specified). This parameter enables hierarchical composition with
                custom mobility. Dictionary with joint parameters as accepted by
                :meth:`ModelBuilder.add_joint` (e.g., joint type, axes, limits, stiffness).

                Cannot be specified together with ``floating``.
            parent_body (int): Parent body index for hierarchical composition. If specified, attaches the
                imported root body to this existing body, making them part of the same kinematic articulation.
                The connection type is determined by ``floating`` or ``base_joint``. If ``-1`` (default),
                the root connects to the world frame. **Restriction**: Only the most recently added
                articulation can be used as parent; attempting to attach to an older articulation will raise
                a ``ValueError``.

                .. note::
                   Valid combinations of ``floating``, ``base_joint``, and ``parent_body``:

                   .. list-table::
                      :header-rows: 1
                      :widths: 15 15 15 55

                      * - floating
                        - base_joint
                        - parent_body
                        - Result
                      * - ``None``
                        - ``None``
                        - ``-1``
                        - Format default (MJCF: honors ``<freejoint>``, else FIXED)
                      * - ``True``
                        - ``None``
                        - ``-1``
                        - FREE joint to world (6 DOF)
                      * - ``False``
                        - ``None``
                        - ``-1``
                        - FIXED joint to world (0 DOF)
                      * - ``None``
                        - ``{dict}``
                        - ``-1``
                        - Custom joint to world (e.g., D6)
                      * - ``False``
                        - ``None``
                        - ``body_idx``
                        - FIXED joint to parent body
                      * - ``None``
                        - ``{dict}``
                        - ``body_idx``
                        - Custom joint to parent body (e.g., D6)
                      * - *explicitly set*
                        - *explicitly set*
                        - *any*
                        - ❌ Error: mutually exclusive (cannot specify both)
                      * - ``True``
                        - ``None``
                        - ``body_idx``
                        - ❌ Error: FREE joints require world frame

            armature_scale (float): Scaling factor to apply to the MJCF-defined joint armature values.
            scale (float): The scaling factor to apply to the imported mechanism.
            hide_visuals (bool): If True, hide visual shapes after loading them (affects visibility, not loading).
            parse_visuals_as_colliders (bool): If True, the geometry defined under the `visual_classes` tags is used for collision handling instead of the `collider_classes` geometries.
            parse_meshes (bool): Whether geometries of type `"mesh"` should be parsed. If False, geometries of type `"mesh"` are ignored.
            parse_sites (bool): Whether sites (non-colliding reference points) should be parsed. If False, sites are ignored.
            parse_visuals (bool): Whether visual geometries (non-collision shapes) should be loaded. If False, visual shapes are not loaded (different from `hide_visuals` which loads but hides them). Default is True.
            parse_mujoco_options (bool): Whether solver options from the MJCF `<option>` tag should be parsed. If False, solver options are not loaded and custom attributes retain their default values. Default is True.
            up_axis (AxisType): The up axis of the MuJoCo scene. The default is Z up.
            ignore_names (Sequence[str]): A list of regular expressions. Bodies and joints with a name matching one of the regular expressions will be ignored.
            ignore_classes (Sequence[str]): A list of regular expressions. Bodies and joints with a class matching one of the regular expressions will be ignored.
            visual_classes (Sequence[str]): A list of regular expressions. Visual geometries with a class matching one of the regular expressions will be parsed.
            collider_classes (Sequence[str]): A list of regular expressions. Collision geometries with a class matching one of the regular expressions will be parsed.
            no_class_as_colliders: If True, geometries without a class are parsed as collision geometries. If False, geometries without a class are parsed as visual geometries.
            force_show_colliders (bool): If True, the collision shapes are always shown, even if there are visual shapes.
            enable_self_collisions (bool): If True, self-collisions are enabled.
            ignore_inertial_definitions (bool): If True, the inertial parameters defined in the MJCF are ignored and the inertia is calculated from the shape geometry.
            ensure_nonstatic_links (bool): If True, links with zero mass are given a small mass (see `static_link_mass`) to ensure they are dynamic.
            static_link_mass (float): The mass to assign to links with zero mass (if `ensure_nonstatic_links` is set to True).
            collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.
            verbose (bool): If True, print additional information about parsing the MJCF.
            skip_equality_constraints (bool): Whether <equality> tags should be parsed. If True, equality constraints are ignored.
            convert_3d_hinge_to_ball_joints (bool): If True, series of three hinge joints are converted to a single ball joint. Default is False.
            mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.
            ctrl_direct (bool): If True, all actuators use :attr:`~newton.solvers.SolverMuJoCo.CtrlSource.CTRL_DIRECT` mode
                where control comes directly from ``control.mujoco.ctrl`` (MuJoCo-native behavior).
                See :ref:`custom_attributes` for details on custom attributes. If False (default), position/velocity
                actuators use :attr:`~newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET` mode where control comes
                from :attr:`newton.Control.joint_target_pos` and :attr:`newton.Control.joint_target_vel`.
            path_resolver (Callable): Callback to resolve file paths. Takes (base_dir, file_path) and returns a resolved path. For <include> elements, can return either a file path or XML content directly. For asset elements (mesh, texture, etc.), must return an absolute file path. The default resolver joins paths and returns absolute file paths.
        """
        from ..solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415
        from ..utils.import_mjcf import parse_mjcf  # noqa: PLC0415

        SolverMuJoCo.register_custom_attributes(self)
        return parse_mjcf(
            self,
            source,
            xform=xform,
            floating=floating,
            base_joint=base_joint,
            parent_body=parent_body,
            armature_scale=armature_scale,
            scale=scale,
            hide_visuals=hide_visuals,
            parse_visuals_as_colliders=parse_visuals_as_colliders,
            parse_meshes=parse_meshes,
            parse_sites=parse_sites,
            parse_visuals=parse_visuals,
            parse_mujoco_options=parse_mujoco_options,
            up_axis=up_axis,
            ignore_names=ignore_names,
            ignore_classes=ignore_classes,
            visual_classes=visual_classes,
            collider_classes=collider_classes,
            no_class_as_colliders=no_class_as_colliders,
            force_show_colliders=force_show_colliders,
            enable_self_collisions=enable_self_collisions,
            ignore_inertial_definitions=ignore_inertial_definitions,
            ensure_nonstatic_links=ensure_nonstatic_links,
            static_link_mass=static_link_mass,
            collapse_fixed_joints=collapse_fixed_joints,
            verbose=verbose,
            skip_equality_constraints=skip_equality_constraints,
            convert_3d_hinge_to_ball_joints=convert_3d_hinge_to_ball_joints,
            mesh_maxhullvert=mesh_maxhullvert,
            ctrl_direct=ctrl_direct,
            path_resolver=path_resolver,
        )

    # endregion

    # region World management methods

    def begin_world(
        self,
        key: str | None = None,
        attributes: dict[str, Any] | None = None,
        gravity: Vec3 | None = None,
    ):
        """Begin a new world context for adding entities.

        This method starts a new world scope where all subsequently added entities
        (bodies, shapes, joints, particles, etc.) will be assigned to this world.
        Use :meth:`end_world` to close the world context and return to the global scope.

        **Important:** Worlds cannot be nested. You must call :meth:`end_world` before
        calling :meth:`begin_world` again.

        Args:
            key (str | None): Optional unique identifier for this world. If None,
                a default key "world_{index}" will be generated.
            attributes (dict[str, Any] | None): Optional custom attributes to associate
                with this world for later use.
            gravity (Vec3 | None): Optional gravity vector for this world. If None,
                the world will use the builder's default gravity (computed from
                ``self.gravity`` and ``self.up_vector``).

        Raises:
            RuntimeError: If called when already inside a world context (current_world != -1).

        Example::

            builder = ModelBuilder()

            # Add global ground plane
            builder.add_ground_plane()  # Added to world -1 (global)

            # Create world 0 with default gravity
            builder.begin_world(key="robot_0")
            builder.add_body(...)  # Added to world 0
            builder.add_shape_box(...)  # Added to world 0
            builder.end_world()

            # Create world 1 with custom zero gravity
            builder.begin_world(key="robot_1", gravity=(0.0, 0.0, 0.0))
            builder.add_body(...)  # Added to world 1
            builder.add_shape_box(...)  # Added to world 1
            builder.end_world()
        """
        if self.current_world != -1:
            raise RuntimeError(
                f"Cannot begin a new world: already in world context (current_world={self.current_world}). "
                "Call end_world() first to close the current world context."
            )

        # Set the current world to the next available world index
        self.current_world = self.num_worlds
        self.num_worlds += 1

        # Store world metadata if needed (for future use)
        # Note: We might want to add world_key and world_attributes lists in __init__ if needed
        # For now, we just track the world index

        # Initialize this world's gravity
        if gravity is not None:
            self.world_gravity.append(tuple(gravity))
        else:
            self.world_gravity.append(tuple(g * self.gravity for g in self.up_vector))

    def end_world(self):
        """End the current world context and return to global scope.

        After calling this method, subsequently added entities will be assigned
        to the global world (-1) until :meth:`begin_world` is called again.

        Raises:
            RuntimeError: If called when not in a world context (current_world == -1).

        Example::

            builder = ModelBuilder()
            builder.begin_world()
            builder.add_body(...)  # Added to current world
            builder.end_world()  # Return to global scope
            builder.add_ground_plane()  # Added to world -1 (global)
        """
        if self.current_world == -1:
            raise RuntimeError("Cannot end world: not currently in a world context (current_world is already -1).")

        # Reset to global world
        self.current_world = -1

    def add_world(self, builder: ModelBuilder, xform: Transform | None = None):
        """Add a builder as a new world.

        This is a convenience method that combines :meth:`begin_world`,
        :meth:`add_builder`, and :meth:`end_world` into a single call.
        It's the recommended way to add homogeneous worlds (multiple instances
        of the same scene/robot).

        Args:
            builder (ModelBuilder): The builder containing entities to add as a new world.
            xform (Transform | None): Optional transform to apply to all root bodies
                in the builder. Useful for spacing out worlds visually.

        Raises:
            RuntimeError: If called when already in a world context (via begin_world).

        Example::

            # Create a robot blueprint
            robot = ModelBuilder()
            robot.add_body(...)
            robot.add_shape_box(...)

            # Create main scene with multiple robot instances
            scene = ModelBuilder()
            scene.add_ground_plane()  # Global ground plane

            # Add multiple robot worlds
            for i in range(3):
                scene.add_world(robot)  # Each robot is a separate world
        """
        self.begin_world()
        self.add_builder(builder, xform=xform)
        self.end_world()

    # endregion

    def add_builder(
        self,
        builder: ModelBuilder,
        xform: Transform | None = None,
    ):
        """Copies the data from another `ModelBuilder` into this `ModelBuilder`.

        All entities from the source builder are added to this builder's current world context
        (the value of `self.current_world`). Any world assignments that existed in the source
        builder are overwritten - all entities will be assigned to the current world.

        Example::

            main_builder = ModelBuilder()
            sub_builder = ModelBuilder()
            sub_builder.add_body(...)
            sub_builder.add_shape_box(...)

            # Adds all entities from sub_builder to main_builder's current world (-1 by default)
            main_builder.add_builder(sub_builder)

            # With transform
            main_builder.add_builder(sub_builder, xform=wp.transform((1, 0, 0)))

        Args:
            builder (ModelBuilder): The model builder to copy data from.
            xform (Transform): Optional offset transform applied to root bodies.
        """

        if builder.up_axis != self.up_axis:
            raise ValueError("Cannot add a builder with a different up axis.")

        # Copy gravity from source builder
        if self.current_world >= 0 and self.current_world < len(self.world_gravity):
            # We're in a world context, update this world's gravity vector
            self.world_gravity[self.current_world] = tuple(g * builder.gravity for g in builder.up_vector)
        elif self.current_world < 0:
            # No world context (add_builder called directly), copy scalar gravity
            self.gravity = builder.gravity

        self._requested_contact_attributes.update(builder._requested_contact_attributes)
        self._requested_state_attributes.update(builder._requested_state_attributes)

        # explicitly resolve the transform multiplication function to avoid
        # repeatedly resolving builtin overloads during shape transformation
        transform_mul_cfunc = wp._src.context.runtime.core.wp_builtin_mul_transformf_transformf

        # dispatches two transform multiplies to the native implementation
        def transform_mul(a, b):
            out = wp.transform.from_buffer(np.empty(7, dtype=np.float32))
            transform_mul_cfunc(a, b, ctypes.byref(out))
            return out

        start_particle_idx = self.particle_count
        start_body_idx = self.body_count
        start_shape_idx = self.shape_count
        start_joint_idx = self.joint_count
        start_joint_dof_idx = self.joint_dof_count
        start_joint_coord_idx = self.joint_coord_count
        start_joint_constraint_idx = self.joint_constraint_count
        start_articulation_idx = self.articulation_count
        start_equality_constraint_idx = len(self.equality_constraint_type)
        start_constraint_mimic_idx = len(self.constraint_mimic_joint0)
        start_edge_idx = self.edge_count
        start_triangle_idx = self.tri_count
        start_tetrahedron_idx = self.tet_count
        start_spring_idx = self.spring_count

        if builder.particle_count:
            self.particle_max_velocity = builder.particle_max_velocity
            if xform is not None:
                pos_offset = xform.p
            else:
                pos_offset = np.zeros(3)
            self.particle_q.extend((np.array(builder.particle_q) + pos_offset).tolist())
            # other particle attributes are added below

        if builder.spring_count:
            self.spring_indices.extend((np.array(builder.spring_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.edge_count:
            # Update edge indices by adding offset, preserving -1 values
            edge_indices = np.array(builder.edge_indices, dtype=np.int32)
            mask = edge_indices != -1
            edge_indices[mask] += start_particle_idx
            self.edge_indices.extend(edge_indices.tolist())
        if builder.tri_count:
            self.tri_indices.extend((np.array(builder.tri_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.tet_count:
            self.tet_indices.extend((np.array(builder.tet_indices, dtype=np.int32) + start_particle_idx).tolist())

        builder_coloring_translated = [group + start_particle_idx for group in builder.particle_color_groups]
        self.particle_color_groups = combine_independent_particle_coloring(
            self.particle_color_groups, builder_coloring_translated
        )

        start_body_idx = self.body_count
        start_shape_idx = self.shape_count
        for s, b in enumerate(builder.shape_body):
            if b > -1:
                new_b = b + start_body_idx
                self.shape_body.append(new_b)
                self.shape_transform.append(builder.shape_transform[s])
            else:
                self.shape_body.append(-1)
                # apply offset transform to root bodies
                if xform is not None:
                    self.shape_transform.append(transform_mul(xform, builder.shape_transform[s]))
                else:
                    self.shape_transform.append(builder.shape_transform[s])

        for b, shapes in builder.body_shapes.items():
            if b == -1:
                self.body_shapes[-1].extend([s + start_shape_idx for s in shapes])
            else:
                self.body_shapes[b + start_body_idx] = [s + start_shape_idx for s in shapes]

        if builder.joint_count:
            start_q = len(self.joint_q)
            start_X_p = len(self.joint_X_p)
            self.joint_X_p.extend(builder.joint_X_p)
            self.joint_q.extend(builder.joint_q)
            if xform is not None:
                for i in range(len(builder.joint_X_p)):
                    if builder.joint_type[i] == JointType.FREE:
                        qi = builder.joint_q_start[i]
                        xform_prev = wp.transform(*builder.joint_q[qi : qi + 7])
                        tf = transform_mul(xform, xform_prev)
                        qi += start_q
                        self.joint_q[qi : qi + 7] = tf
                    elif builder.joint_parent[i] == -1:
                        self.joint_X_p[start_X_p + i] = transform_mul(xform, builder.joint_X_p[i])

            # offset the indices
            self.articulation_start.extend([a + self.joint_count for a in builder.articulation_start])

            new_parents = [p + start_body_idx if p != -1 else -1 for p in builder.joint_parent]
            new_children = [c + start_body_idx for c in builder.joint_child]

            self.joint_parent.extend(new_parents)
            self.joint_child.extend(new_children)

            # Update parent/child lookups
            for p, c in zip(new_parents, new_children, strict=True):
                if c not in self.joint_parents:
                    self.joint_parents[c] = [p]
                else:
                    self.joint_parents[c].append(p)

                if p not in self.joint_children:
                    self.joint_children[p] = [c]
                elif c not in self.joint_children[p]:
                    self.joint_children[p].append(c)

            self.joint_q_start.extend([c + self.joint_coord_count for c in builder.joint_q_start])
            self.joint_qd_start.extend([c + self.joint_dof_count for c in builder.joint_qd_start])
            self.joint_cts_start.extend([c + self.joint_constraint_count for c in builder.joint_cts_start])

        if xform is not None:
            for i in range(builder.body_count):
                self.body_q.append(transform_mul(xform, builder.body_q[i]))
        else:
            self.body_q.extend(builder.body_q)

        # Copy collision groups without modification
        self.shape_collision_group.extend(builder.shape_collision_group)

        # Copy collision filter pairs with offset
        self.shape_collision_filter_pairs.extend(
            [(i + start_shape_idx, j + start_shape_idx) for i, j in builder.shape_collision_filter_pairs]
        )

        # Handle world assignments
        # For particles
        if builder.particle_count > 0:
            # Override all world indices with current world
            particle_groups = [self.current_world] * builder.particle_count
            self.particle_world.extend(particle_groups)

        # For bodies
        if builder.body_count > 0:
            body_groups = [self.current_world] * builder.body_count
            self.body_world.extend(body_groups)

        # For shapes
        if builder.shape_count > 0:
            shape_worlds = [self.current_world] * builder.shape_count
            self.shape_world.extend(shape_worlds)

        # For joints
        if builder.joint_count > 0:
            s = [self.current_world] * builder.joint_count
            self.joint_world.extend(s)
            # Offset articulation indices for joints (-1 stays -1)
            self.joint_articulation.extend(
                [a + start_articulation_idx if a >= 0 else -1 for a in builder.joint_articulation]
            )

        # For articulations
        if builder.articulation_count > 0:
            articulation_groups = [self.current_world] * builder.articulation_count
            self.articulation_world.extend(articulation_groups)

        # For equality constraints
        if len(builder.equality_constraint_type) > 0:
            constraint_worlds = [self.current_world] * len(builder.equality_constraint_type)
            self.equality_constraint_world.extend(constraint_worlds)

            # Remap body and joint indices in equality constraints
            self.equality_constraint_type.extend(builder.equality_constraint_type)
            self.equality_constraint_body1.extend(
                [b + start_body_idx if b != -1 else -1 for b in builder.equality_constraint_body1]
            )
            self.equality_constraint_body2.extend(
                [b + start_body_idx if b != -1 else -1 for b in builder.equality_constraint_body2]
            )
            self.equality_constraint_anchor.extend(builder.equality_constraint_anchor)
            self.equality_constraint_torquescale.extend(builder.equality_constraint_torquescale)
            self.equality_constraint_relpose.extend(builder.equality_constraint_relpose)
            self.equality_constraint_joint1.extend(
                [j + start_joint_idx if j != -1 else -1 for j in builder.equality_constraint_joint1]
            )
            self.equality_constraint_joint2.extend(
                [j + start_joint_idx if j != -1 else -1 for j in builder.equality_constraint_joint2]
            )
            self.equality_constraint_polycoef.extend(builder.equality_constraint_polycoef)
            self.equality_constraint_key.extend(builder.equality_constraint_key)
            self.equality_constraint_enabled.extend(builder.equality_constraint_enabled)

        # For mimic constraints
        if len(builder.constraint_mimic_joint0) > 0:
            constraint_worlds = [self.current_world] * len(builder.constraint_mimic_joint0)
            self.constraint_mimic_world.extend(constraint_worlds)

            # Remap joint indices in mimic constraints
            self.constraint_mimic_joint0.extend(
                [j + start_joint_idx if j != -1 else -1 for j in builder.constraint_mimic_joint0]
            )
            self.constraint_mimic_joint1.extend(
                [j + start_joint_idx if j != -1 else -1 for j in builder.constraint_mimic_joint1]
            )
            self.constraint_mimic_coef0.extend(builder.constraint_mimic_coef0)
            self.constraint_mimic_coef1.extend(builder.constraint_mimic_coef1)
            self.constraint_mimic_enabled.extend(builder.constraint_mimic_enabled)
            self.constraint_mimic_key.extend(builder.constraint_mimic_key)

        more_builder_attrs = [
            "articulation_key",
            "body_inertia",
            "body_mass",
            "body_inv_inertia",
            "body_inv_mass",
            "body_com",
            "body_lock_inertia",
            "body_qd",
            "body_key",
            "joint_type",
            "joint_enabled",
            "joint_X_c",
            "joint_armature",
            "joint_axis",
            "joint_dof_dim",
            "joint_key",
            "joint_qd",
            "joint_cts",
            "joint_f",
            "joint_target_pos",
            "joint_target_vel",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
            "joint_act_mode",
            "joint_effort_limit",
            "joint_velocity_limit",
            "joint_friction",
            "shape_key",
            "shape_flags",
            "shape_type",
            "shape_scale",
            "shape_source",
            "shape_is_solid",
            "shape_thickness",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_ka",
            "shape_material_mu",
            "shape_material_restitution",
            "shape_material_torsional_friction",
            "shape_material_rolling_friction",
            "shape_material_k_hydro",
            "shape_collision_radius",
            "shape_contact_margin",
            "shape_sdf_narrow_band_range",
            "shape_sdf_max_resolution",
            "shape_sdf_target_voxel_size",
            "particle_qd",
            "particle_mass",
            "particle_radius",
            "particle_flags",
            "edge_rest_angle",
            "edge_rest_length",
            "edge_bending_properties",
            "spring_rest_length",
            "spring_stiffness",
            "spring_damping",
            "spring_control",
            "tri_poses",
            "tri_activations",
            "tri_materials",
            "tri_areas",
            "tet_poses",
            "tet_activations",
            "tet_materials",
        ]

        for attr in more_builder_attrs:
            getattr(self, attr).extend(getattr(builder, attr))

        self.joint_dof_count += builder.joint_dof_count
        self.joint_coord_count += builder.joint_coord_count
        self.joint_constraint_count += builder.joint_constraint_count

        # Merge custom attributes from the sub-builder
        # Shared offset map for both frequency and references
        # Note: "world" is NOT included here - WORLD frequency is handled specially
        entity_offsets = {
            "body": start_body_idx,
            "shape": start_shape_idx,
            "joint": start_joint_idx,
            "joint_dof": start_joint_dof_idx,
            "joint_coord": start_joint_coord_idx,
            "joint_constraint": start_joint_constraint_idx,
            "articulation": start_articulation_idx,
            "equality_constraint": start_equality_constraint_idx,
            "constraint_mimic": start_constraint_mimic_idx,
            "particle": start_particle_idx,
            "edge": start_edge_idx,
            "triangle": start_triangle_idx,
            "tetrahedron": start_tetrahedron_idx,
            "spring": start_spring_idx,
        }

        # Snapshot custom frequency counts BEFORE iteration (they get updated during merge)
        custom_frequency_offsets = dict(self._custom_frequency_counts)

        def get_offset(entity_or_key: str | None) -> int:
            """Get offset for an entity type or custom frequency."""
            if entity_or_key is None:
                return 0
            if entity_or_key in entity_offsets:
                return entity_offsets[entity_or_key]
            if entity_or_key in custom_frequency_offsets:
                return custom_frequency_offsets[entity_or_key]
            if entity_or_key in builder._custom_frequency_counts:
                return 0
            raise ValueError(
                f"Unknown references value '{entity_or_key}'. "
                f"Valid values are: {list(entity_offsets.keys())} or custom frequencies."
            )

        for full_key, attr in builder.custom_attributes.items():
            # Fast path: skip attributes with no values (avoids computing offsets/closures)
            if not attr.values:
                # Still need to declare empty attribute on first merge
                if full_key not in self.custom_attributes:
                    freq_key = attr.frequency_key
                    mapped_values = [] if isinstance(freq_key, str) else {}
                    self.custom_attributes[full_key] = replace(attr, values=mapped_values)
                continue

            # Index offset based on frequency
            freq_key = attr.frequency_key
            if isinstance(freq_key, str):
                # Custom frequency: offset by pre-merge count
                index_offset = custom_frequency_offsets.get(freq_key, 0)
            elif attr.frequency == Model.AttributeFrequency.ONCE:
                index_offset = 0
            elif attr.frequency == Model.AttributeFrequency.WORLD:
                # WORLD frequency: indices are keyed by world index, not by offset
                # When called via add_world(), current_world is the world being added
                index_offset = 0 if self.current_world == -1 else self.current_world
            else:
                index_offset = get_offset(attr.frequency.name.lower())

            # Value transformation based on references
            use_current_world = attr.references == "world"
            value_offset = 0 if use_current_world else get_offset(attr.references)

            def transform_value(v, offset=value_offset, replace_with_world=use_current_world):
                if replace_with_world:
                    return self.current_world
                if offset == 0:
                    return v
                # Handle integers, preserving negative sentinels (e.g., -1 means "invalid")
                if isinstance(v, int):
                    return v + offset if v >= 0 else v
                # Handle list/tuple explicitly, preserving negative sentinels in elements
                if isinstance(v, (list, tuple)):
                    transformed = [x + offset if isinstance(x, int) and x >= 0 else x for x in v]
                    return type(v)(transformed)
                # For other types (numpy, warp, etc.), try arithmetic offset
                try:
                    return v + offset
                except TypeError:
                    return v

            # Declare the attribute if it doesn't exist in the main builder
            merged = self.custom_attributes.get(full_key)
            if merged is None:
                if isinstance(freq_key, str):
                    # String frequency: copy list as-is (no offset for sequential data)
                    mapped_values = [transform_value(value) for value in attr.values]
                else:
                    # Enum frequency: remap dict indices with offset
                    mapped_values = {index_offset + idx: transform_value(value) for idx, value in attr.values.items()}
                self.custom_attributes[full_key] = replace(attr, values=mapped_values)
                continue

            # Prevent silent divergence if defaults differ
            # Handle array/vector types by converting to comparable format
            try:
                defaults_match = merged.default == attr.default
                # Handle array-like comparisons
                if hasattr(defaults_match, "__iter__") and not isinstance(defaults_match, (str, bytes)):
                    defaults_match = all(defaults_match)
            except (ValueError, TypeError):
                # If comparison fails, assume they're different
                defaults_match = False

            if not defaults_match:
                raise ValueError(
                    f"Custom attribute '{full_key}' default mismatch when merging builders: "
                    f"existing={merged.default}, incoming={attr.default}"
                )

            # Remap indices and copy values
            if merged.values is None:
                merged.values = [] if isinstance(freq_key, str) else {}

            if isinstance(freq_key, str):
                # String frequency: extend list with transformed values
                new_values = [transform_value(value) for value in attr.values]
                merged.values.extend(new_values)
            else:
                # Enum frequency: update dict with remapped indices
                new_indices = {index_offset + idx: transform_value(value) for idx, value in attr.values.items()}
                merged.values.update(new_indices)

        # Update custom frequency counts once per unique frequency (not per attribute)
        for freq_key, builder_count in builder._custom_frequency_counts.items():
            offset = custom_frequency_offsets.get(freq_key, 0)
            self._custom_frequency_counts[freq_key] = offset + builder_count

    @staticmethod
    def _coerce_mat33(value: Any) -> wp.mat33:
        """Coerce a mat33-like value into a wp.mat33 without triggering Warp row-vector constructor warnings."""
        if wp.types.type_is_matrix(type(value)):
            return value

        if isinstance(value, (list, tuple)) and len(value) == 3:
            rows = []
            is_rows = True
            for r in value:
                if wp.types.type_is_vector(type(r)):
                    rows.append(wp.vec3(*r))
                elif isinstance(r, (list, tuple, np.ndarray)) and len(r) == 3:
                    rows.append(wp.vec3(*r))
                else:
                    is_rows = False
                    break
            if is_rows:
                return wp.matrix_from_rows(*rows)

        if isinstance(value, np.ndarray) and value.shape == (3, 3):
            return wp.mat33(*value.reshape(-1).tolist())

        return wp.mat33(*value)

    def add_link(
        self,
        xform: Transform | None = None,
        armature: float | None = None,
        com: Vec3 | None = None,
        inertia: Mat33 | None = None,
        mass: float = 0.0,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
        lock_inertia: bool = False,
    ) -> int:
        """Adds a link (rigid body) to the model within an articulation.

        This method creates a link without automatically adding a joint. To connect this link
        to the articulation structure, you must explicitly call one of the joint methods
        (e.g., :meth:`add_joint_revolute`, :meth:`add_joint_fixed`, etc.) after creating the link.

        After calling this method and one of the joint methods, ensure that an articulation is created using :meth:`add_articulation`.

        Args:
            xform: The location of the body in the world frame.
            armature: Artificial inertia added to the body. If None, the default value from :attr:`default_body_armature` is used.
            com: The center of mass of the body w.r.t its origin. If None, the center of mass is assumed to be at the origin.
            inertia: The 3x3 inertia tensor of the body (specified relative to the center of mass). If None, the inertia tensor is assumed to be zero.
            mass: Mass of the body.
            key: Key of the body (optional).
            custom_attributes: Dictionary of custom attribute names to values.
            lock_inertia: If True, prevents subsequent shape additions from modifying this body's mass,
                center of mass, or inertia. This does not affect merging behavior in
                :meth:`collapse_fixed_joints`, which always accumulates mass and inertia across merged bodies.

        Returns:
            The index of the body in the model.

        Note:
            If the mass is zero then the body is treated as kinematic with no dynamics.

        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        if com is None:
            com = wp.vec3()
        else:
            com = wp.vec3(*com)
        if inertia is None:
            inertia = wp.mat33()
        else:
            inertia = self._coerce_mat33(inertia)

        body_id = len(self.body_mass)

        # body data
        if armature is None:
            armature = self.default_body_armature
        inertia = inertia + wp.mat33(np.eye(3, dtype=np.float32)) * armature
        self.body_inertia.append(inertia)
        self.body_mass.append(mass)
        self.body_com.append(com)
        self.body_lock_inertia.append(lock_inertia)

        if mass > 0.0:
            self.body_inv_mass.append(1.0 / mass)
        else:
            self.body_inv_mass.append(0.0)

        if any(x for x in inertia):
            self.body_inv_inertia.append(wp.inverse(inertia))
        else:
            self.body_inv_inertia.append(inertia)

        self.body_q.append(xform)
        self.body_qd.append(wp.spatial_vector())

        self.body_key.append(key or f"body_{body_id}")
        self.body_shapes[body_id] = []
        self.body_world.append(self.current_world)
        # Process custom attributes
        if custom_attributes:
            self._process_custom_attributes(
                entity_index=body_id,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.BODY,
            )

        return body_id

    def add_body(
        self,
        xform: Transform | None = None,
        armature: float | None = None,
        com: Vec3 | None = None,
        inertia: Mat33 | None = None,
        mass: float = 0.0,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
        lock_inertia: bool = False,
    ) -> int:
        """Adds a stand-alone free-floating rigid body to the model.

        This is a convenience method that creates a single-body articulation with a free joint,
        allowing the body to move freely in 6 degrees of freedom. Internally, this method calls:

        1. :meth:`add_link` to create the body
        2. :meth:`add_joint_free` to add a free joint connecting the body to the world
        3. :meth:`add_articulation` to create a new articulation from the joint

        For creating articulations with multiple linked bodies, use :meth:`add_link`,
        the appropriate joint methods, and :meth:`add_articulation` directly.

        Args:
            xform: The location of the body in the world frame.
            armature: Artificial inertia added to the body. If None, the default value from :attr:`default_body_armature` is used.
            com: The center of mass of the body w.r.t its origin. If None, the center of mass is assumed to be at the origin.
            inertia: The 3x3 inertia tensor of the body (specified relative to the center of mass). If None, the inertia tensor is assumed to be zero.
            mass: Mass of the body.
            key: Key of the body. When provided, the auto-created free joint and articulation
                are assigned keys ``{key}_free_joint`` and ``{key}_articulation`` respectively.
            custom_attributes: Dictionary of custom attribute names to values.
            lock_inertia: If True, prevents subsequent shape additions from modifying this body's mass,
                center of mass, or inertia. This does not affect merging behavior in
                :meth:`collapse_fixed_joints`, which always accumulates mass and inertia across merged bodies.

        Returns:
            The index of the body in the model.

        Note:
            If the mass is zero then the body is treated as kinematic with no dynamics.

        """
        # Create the link
        body_id = self.add_link(
            xform=xform,
            armature=armature,
            com=com,
            inertia=inertia,
            mass=mass,
            key=key,
            custom_attributes=custom_attributes,
            lock_inertia=lock_inertia,
        )

        # Add a free joint to make it float
        joint_id = self.add_joint_free(
            child=body_id,
            key=f"{key}_free_joint" if key else None,
        )

        # Create an articulation from the joint
        articulation_key = f"{key}_articulation" if key else None
        self.add_articulation([joint_id], key=articulation_key)

        return body_id

    # region joints

    def add_joint(
        self,
        joint_type: JointType,
        parent: int,
        child: int,
        linear_axes: list[JointDofConfig] | None = None,
        angular_axes: list[JointDofConfig] | None = None,
        key: str | None = None,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """
        Generic method to add any type of joint to this ModelBuilder.

        Args:
            joint_type (JointType): The type of joint to add (see :ref:`Joint types`).
            parent (int): The index of the parent body (-1 is the world).
            child (int): The index of the child body.
            linear_axes (list(:class:`JointDofConfig`)): The linear axes (see :class:`JointDofConfig`) of the joint,
                defined in the joint parent anchor frame.
            angular_axes (list(:class:`JointDofConfig`)): The angular axes (see :class:`JointDofConfig`) of the joint,
                defined in the joint parent anchor frame.
            key (str): The key of the joint (optional).
            parent_xform (Transform): The transform from the parent body frame to the joint parent anchor frame.
                If None, the identity transform is used.
            child_xform (Transform): The transform from the child body frame to the joint child anchor frame.
                If None, the identity transform is used.
            collision_filter_parent (bool): Whether to filter collisions between shapes of the parent and child bodies.
            enabled (bool): Whether the joint is enabled (not considered by :class:`SolverFeatherstone`).
            custom_attributes: Dictionary of custom attribute keys (see :attr:`CustomAttribute.key`) to values. Note that custom attributes with frequency :attr:`Model.AttributeFrequency.JOINT_DOF` or :attr:`Model.AttributeFrequency.JOINT_COORD` can be provided as: (1) lists with length equal to the joint's DOF or coordinate count, (2) dicts mapping DOF/coordinate indices to values, or (3) scalar values for single-DOF/single-coordinate joints (automatically expanded to lists). Custom attributes with frequency :attr:`Model.AttributeFrequency.JOINT` require a single value to be defined.

        Returns:
            The index of the added joint.
        """
        if linear_axes is None:
            linear_axes = []
        if angular_axes is None:
            angular_axes = []

        if parent_xform is None:
            parent_xform = wp.transform()
        else:
            parent_xform = wp.transform(*parent_xform)
        if child_xform is None:
            child_xform = wp.transform()
        else:
            child_xform = wp.transform(*child_xform)

        # Validate that parent and child bodies belong to the current world
        if parent != -1:  # -1 means world/ground
            if parent < 0 or parent >= len(self.body_world):
                raise ValueError(f"Parent body index {parent} is out of range")
            if self.body_world[parent] != self.current_world:
                raise ValueError(
                    f"Cannot create joint: parent body {parent} belongs to world {self.body_world[parent]}, "
                    f"but current world is {self.current_world}"
                )

        if child < 0 or child >= len(self.body_world):
            raise ValueError(f"Child body index {child} is out of range")
        if self.body_world[child] != self.current_world:
            raise ValueError(
                f"Cannot create joint: child body {child} belongs to world {self.body_world[child]}, "
                f"but current world is {self.current_world}"
            )

        self.joint_type.append(joint_type)
        self.joint_parent.append(parent)
        if child not in self.joint_parents:
            self.joint_parents[child] = [parent]
        else:
            self.joint_parents[child].append(parent)
        if parent not in self.joint_children:
            self.joint_children[parent] = [child]
        elif child not in self.joint_children[parent]:
            self.joint_children[parent].append(child)
        self.joint_child.append(child)
        self.joint_X_p.append(wp.transform(parent_xform))
        self.joint_X_c.append(wp.transform(child_xform))
        self.joint_key.append(key or f"joint_{self.joint_count}")
        self.joint_dof_dim.append((len(linear_axes), len(angular_axes)))
        self.joint_enabled.append(enabled)
        self.joint_world.append(self.current_world)
        self.joint_articulation.append(-1)

        def add_axis_dim(dim: ModelBuilder.JointDofConfig):
            self.joint_axis.append(dim.axis)
            self.joint_target_pos.append(dim.target_pos)
            self.joint_target_vel.append(dim.target_vel)

            # Use actuator_mode if explicitly set, otherwise infer from gains
            if dim.actuator_mode is not None:
                mode = int(dim.actuator_mode)
            else:
                # Infer has_drive from whether gains are non-zero: non-zero gains imply a drive exists.
                # This ensures freejoints (gains=0) get NONE, while joints with gains get appropriate mode.
                has_drive = dim.target_ke != 0.0 or dim.target_kd != 0.0
                mode = int(ActuatorMode.from_gains(dim.target_ke, dim.target_kd, has_drive=has_drive))

            # Store per-DOF actuator properties
            self.joint_act_mode.append(mode)
            self.joint_target_ke.append(dim.target_ke)
            self.joint_target_kd.append(dim.target_kd)
            self.joint_limit_ke.append(dim.limit_ke)
            self.joint_limit_kd.append(dim.limit_kd)
            self.joint_armature.append(dim.armature)
            self.joint_effort_limit.append(dim.effort_limit)
            self.joint_velocity_limit.append(dim.velocity_limit)
            self.joint_friction.append(dim.friction)
            if np.isfinite(dim.limit_lower):
                self.joint_limit_lower.append(dim.limit_lower)
            else:
                self.joint_limit_lower.append(-MAXVAL)
            if np.isfinite(dim.limit_upper):
                self.joint_limit_upper.append(dim.limit_upper)
            else:
                self.joint_limit_upper.append(MAXVAL)

        for dim in linear_axes:
            add_axis_dim(dim)
        for dim in angular_axes:
            add_axis_dim(dim)

        dof_count, coord_count = joint_type.dof_count(len(linear_axes) + len(angular_axes))
        cts_count = joint_type.constraint_count(len(linear_axes) + len(angular_axes))

        for _ in range(coord_count):
            self.joint_q.append(0.0)
        for _ in range(dof_count):
            self.joint_qd.append(0.0)
            self.joint_f.append(0.0)
        for _ in range(cts_count):
            self.joint_cts.append(0.0)

        if joint_type == JointType.FREE or joint_type == JointType.DISTANCE or joint_type == JointType.BALL:
            # ensure that a valid quaternion is used for the angular dofs
            self.joint_q[-1] = 1.0

        self.joint_q_start.append(self.joint_coord_count)
        self.joint_qd_start.append(self.joint_dof_count)
        self.joint_cts_start.append(self.joint_constraint_count)

        self.joint_dof_count += dof_count
        self.joint_coord_count += coord_count
        self.joint_constraint_count += cts_count

        if collision_filter_parent and parent > -1:
            for child_shape in self.body_shapes[child]:
                if not self.shape_flags[child_shape] & ShapeFlags.COLLIDE_SHAPES:
                    continue
                for parent_shape in self.body_shapes[parent]:
                    if not self.shape_flags[parent_shape] & ShapeFlags.COLLIDE_SHAPES:
                        continue
                    self.add_shape_collision_filter_pair(parent_shape, child_shape)

        joint_index = self.joint_count - 1

        # Process custom attributes
        if custom_attributes:
            self._process_joint_custom_attributes(
                joint_index=joint_index,
                custom_attrs=custom_attributes,
            )

        return joint_index

    def add_joint_revolute(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        axis: AxisType | Vec3 | JointDofConfig | None = None,
        target_pos: float | None = None,
        target_vel: float | None = None,
        target_ke: float | None = None,
        target_kd: float | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        limit_ke: float | None = None,
        limit_kd: float | None = None,
        armature: float | None = None,
        effort_limit: float | None = None,
        velocity_limit: float | None = None,
        friction: float | None = None,
        actuator_mode: ActuatorMode | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
        **kwargs,
    ) -> int:
        """Adds a revolute (hinge) joint to the model. It has one degree of freedom.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform from the parent body frame to the joint parent anchor frame.
            child_xform (Transform): The transform from the child body frame to the joint child anchor frame.
            axis (AxisType | Vec3 | JointDofConfig): The axis of rotation in the joint parent anchor frame, which is
                the parent body's local frame transformed by `parent_xform`. It can be a :class:`JointDofConfig` object
                whose settings will be used instead of the other arguments.
            target_pos: The target position of the joint.
            target_vel: The target velocity of the joint.
            target_ke: The stiffness of the joint target.
            target_kd: The damping of the joint target.
            limit_lower: The lower limit of the joint. If None, the default value from :attr:`default_joint_limit_lower` is used.
            limit_upper: The upper limit of the joint. If None, the default value from :attr:`default_joint_limit_upper` is used.
            limit_ke: The stiffness of the joint limit. If None, the default value from :attr:`default_joint_limit_ke` is used.
            limit_kd: The damping of the joint limit. If None, the default value from :attr:`default_joint_limit_kd` is used.
            armature: Artificial inertia added around the joint axis. If None, the default value from :attr:`default_joint_armature` is used.
            effort_limit: Maximum effort (force/torque) the joint axis can exert. If None, the default value from :attr:`default_joint_cfg.effort_limit` is used.
            velocity_limit: Maximum velocity the joint axis can achieve. If None, the default value from :attr:`default_joint_cfg.velocity_limit` is used.
            friction: Friction coefficient for the joint axis. If None, the default value from :attr:`default_joint_cfg.friction` is used.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.
            custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD frequency attributes.

        Returns:
            The index of the added joint.

        """

        if axis is None:
            axis = self.default_joint_cfg.axis
        if isinstance(axis, ModelBuilder.JointDofConfig):
            ax = axis
        else:
            ax = ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=limit_lower if limit_lower is not None else self.default_joint_cfg.limit_lower,
                limit_upper=limit_upper if limit_upper is not None else self.default_joint_cfg.limit_upper,
                target_pos=target_pos if target_pos is not None else self.default_joint_cfg.target_pos,
                target_vel=target_vel if target_vel is not None else self.default_joint_cfg.target_vel,
                target_ke=target_ke if target_ke is not None else self.default_joint_cfg.target_ke,
                target_kd=target_kd if target_kd is not None else self.default_joint_cfg.target_kd,
                limit_ke=limit_ke if limit_ke is not None else self.default_joint_cfg.limit_ke,
                limit_kd=limit_kd if limit_kd is not None else self.default_joint_cfg.limit_kd,
                armature=armature if armature is not None else self.default_joint_cfg.armature,
                effort_limit=effort_limit if effort_limit is not None else self.default_joint_cfg.effort_limit,
                velocity_limit=velocity_limit if velocity_limit is not None else self.default_joint_cfg.velocity_limit,
                friction=friction if friction is not None else self.default_joint_cfg.friction,
                actuator_mode=actuator_mode if actuator_mode is not None else self.default_joint_cfg.actuator_mode,
            )
        return self.add_joint(
            JointType.REVOLUTE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            angular_axes=[ax],
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            custom_attributes=custom_attributes,
            **kwargs,
        )

    def add_joint_prismatic(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        axis: AxisType | Vec3 | JointDofConfig = Axis.X,
        target_pos: float | None = None,
        target_vel: float | None = None,
        target_ke: float | None = None,
        target_kd: float | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        limit_ke: float | None = None,
        limit_kd: float | None = None,
        armature: float | None = None,
        effort_limit: float | None = None,
        velocity_limit: float | None = None,
        friction: float | None = None,
        actuator_mode: ActuatorMode | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a prismatic (sliding) joint to the model. It has one degree of freedom.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform from the parent body frame to the joint parent anchor frame.
            child_xform (Transform): The transform from the child body frame to the joint child anchor frame.
            axis (AxisType | Vec3 | JointDofConfig): The axis of translation in the joint parent anchor frame, which is
                the parent body's local frame transformed by `parent_xform`. It can be a :class:`JointDofConfig` object
                whose settings will be used instead of the other arguments.
            target_pos: The target position of the joint.
            target_vel: The target velocity of the joint.
            target_ke: The stiffness of the joint target.
            target_kd: The damping of the joint target.
            limit_lower: The lower limit of the joint. If None, the default value from :attr:`default_joint_limit_lower` is used.
            limit_upper: The upper limit of the joint. If None, the default value from :attr:`default_joint_limit_upper` is used.
            limit_ke: The stiffness of the joint limit. If None, the default value from :attr:`default_joint_limit_ke` is used.
            limit_kd: The damping of the joint limit. If None, the default value from :attr:`default_joint_limit_kd` is used.
            armature: Artificial inertia added around the joint axis. If None, the default value from :attr:`default_joint_armature` is used.
            effort_limit: Maximum effort (force) the joint axis can exert. If None, the default value from :attr:`default_joint_cfg.effort_limit` is used.
            velocity_limit: Maximum velocity the joint axis can achieve. If None, the default value from :attr:`default_joint_cfg.velocity_limit` is used.
            friction: Friction coefficient for the joint axis. If None, the default value from :attr:`default_joint_cfg.friction` is used.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.
            custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD frequency attributes.

        Returns:
            The index of the added joint.

        """

        if axis is None:
            axis = self.default_joint_cfg.axis
        if isinstance(axis, ModelBuilder.JointDofConfig):
            ax = axis
        else:
            ax = ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=limit_lower if limit_lower is not None else self.default_joint_cfg.limit_lower,
                limit_upper=limit_upper if limit_upper is not None else self.default_joint_cfg.limit_upper,
                target_pos=target_pos if target_pos is not None else self.default_joint_cfg.target_pos,
                target_vel=target_vel if target_vel is not None else self.default_joint_cfg.target_vel,
                target_ke=target_ke if target_ke is not None else self.default_joint_cfg.target_ke,
                target_kd=target_kd if target_kd is not None else self.default_joint_cfg.target_kd,
                limit_ke=limit_ke if limit_ke is not None else self.default_joint_cfg.limit_ke,
                limit_kd=limit_kd if limit_kd is not None else self.default_joint_cfg.limit_kd,
                armature=armature if armature is not None else self.default_joint_cfg.armature,
                effort_limit=effort_limit if effort_limit is not None else self.default_joint_cfg.effort_limit,
                velocity_limit=velocity_limit if velocity_limit is not None else self.default_joint_cfg.velocity_limit,
                friction=friction if friction is not None else self.default_joint_cfg.friction,
                actuator_mode=actuator_mode if actuator_mode is not None else self.default_joint_cfg.actuator_mode,
            )
        return self.add_joint(
            JointType.PRISMATIC,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[ax],
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            custom_attributes=custom_attributes,
        )

    def add_joint_ball(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        armature: float | None = None,
        friction: float | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
        actuator_mode: ActuatorMode | None = None,
    ) -> int:
        """Adds a ball (spherical) joint to the model. Its position is defined by a 4D quaternion (xyzw) and its velocity is a 3D vector.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform from the parent body frame to the joint parent anchor frame.
            child_xform (Transform): The transform from the child body frame to the joint child anchor frame.
            armature: Artificial inertia added around the joint axes. If None, the default value from :attr:`default_joint_armature` is used.
            friction: Friction coefficient for the joint axes. If None, the default value from :attr:`default_joint_cfg.friction` is used.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.
            custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD frequency attributes.
            actuator_mode: The actuator mode for this joint's DOFs. If None, defaults to NONE.

        Returns:
            The index of the added joint.

        .. note:: Target position and velocity control for ball joints is currently only supported in :class:`newton.solvers.SolverMuJoCo`.

        """

        if armature is None:
            armature = self.default_joint_cfg.armature
        if friction is None:
            friction = self.default_joint_cfg.friction

        x = ModelBuilder.JointDofConfig(
            axis=Axis.X,
            armature=armature,
            friction=friction,
            actuator_mode=actuator_mode,
        )
        y = ModelBuilder.JointDofConfig(
            axis=Axis.Y,
            armature=armature,
            friction=friction,
            actuator_mode=actuator_mode,
        )
        z = ModelBuilder.JointDofConfig(
            axis=Axis.Z,
            armature=armature,
            friction=friction,
            actuator_mode=actuator_mode,
        )

        return self.add_joint(
            JointType.BALL,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            angular_axes=[x, y, z],
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            custom_attributes=custom_attributes,
        )

    def add_joint_fixed(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a fixed (static) joint to the model. It has no degrees of freedom.
        See :meth:`collapse_fixed_joints` for a helper function that removes these fixed joints and merges the connecting bodies to simplify the model and improve stability.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.
            custom_attributes: Dictionary of custom attribute values for JOINT frequency attributes.

        Returns:
            The index of the added joint

        """

        joint_index = self.add_joint(
            JointType.FIXED,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

        # Process custom attributes (only JOINT frequency is valid for fixed joints)
        if custom_attributes:
            self._process_joint_custom_attributes(joint_index, custom_attributes)

        return joint_index

    def add_joint_free(
        self,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        parent: int = -1,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a free joint to the model.
        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation) and 6 velocity degrees of freedom (see :ref:`Twist conventions in Newton <Twist conventions>`).
        The positional dofs are initialized by the child body's transform (see :attr:`body_q` and the ``xform`` argument to :meth:`add_body`).

        Args:
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            parent: The index of the parent body (-1 by default to use the world frame, e.g. to make the child body and its children a floating-base mechanism).
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.
            custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD frequency attributes.

        Returns:
            The index of the added joint.

        """

        joint_id = self.add_joint(
            JointType.FREE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            linear_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            angular_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            custom_attributes=custom_attributes,
        )
        q_start = self.joint_q_start[joint_id]
        # set the positional dofs to the child body's transform
        self.joint_q[q_start : q_start + 7] = list(self.body_q[child])
        return joint_id

    def add_joint_distance(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        min_distance: float = -1.0,
        max_distance: float = 1.0,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a distance joint to the model. The distance joint constraints the distance between the joint anchor points on the two bodies (see :ref:`FK-IK`) it connects to the interval [`min_distance`, `max_distance`].
        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation) and 6 velocity degrees of freedom (first 3 linear and then 3 angular velocity dimensions).

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            min_distance: The minimum distance between the bodies (no limit if negative).
            max_distance: The maximum distance between the bodies (no limit if negative).
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.
            custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD frequency attributes.

        Returns:
            The index of the added joint.

        .. note:: Distance joints are currently only supported in :class:`newton.solvers.SolverXPBD`.

        """

        ax = ModelBuilder.JointDofConfig(
            axis=(1.0, 0.0, 0.0),
            limit_lower=min_distance,
            limit_upper=max_distance,
        )
        return self.add_joint(
            JointType.DISTANCE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[
                ax,
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            angular_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            custom_attributes=custom_attributes,
        )

    def add_joint_d6(
        self,
        parent: int,
        child: int,
        linear_axes: Sequence[JointDofConfig] | None = None,
        angular_axes: Sequence[JointDofConfig] | None = None,
        key: str | None = None,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
        **kwargs,
    ) -> int:
        """Adds a generic joint with custom linear and angular axes. The number of axes determines the number of degrees of freedom of the joint.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            linear_axes: A list of linear axes.
            angular_axes: A list of angular axes.
            key: The key of the joint.
            parent_xform (Transform): The transform from the parent body frame to the joint parent anchor frame.
            child_xform (Transform): The transform from the child body frame to the joint child anchor frame.
            armature: Artificial inertia added around the joint axes. If None, the default value from :attr:`default_joint_armature` is used.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.
            custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD frequency attributes.

        Returns:
            The index of the added joint.

        """
        if linear_axes is None:
            linear_axes = []
        if angular_axes is None:
            angular_axes = []

        return self.add_joint(
            JointType.D6,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=list(linear_axes),
            angular_axes=list(angular_axes),
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            custom_attributes=custom_attributes,
            **kwargs,
        )

    def add_joint_cable(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        stretch_stiffness: float | None = None,
        stretch_damping: float | None = None,
        bend_stiffness: float | None = None,
        bend_damping: float | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
        **kwargs,
    ) -> int:
        """Adds a cable joint to the model. It has two degrees of freedom: one linear (stretch)
        that constrains the distance between the attachment points, and one angular (bend/twist)
        that penalizes the relative rotation of the attachment frames.

        .. note::

            Cable joints are supported by :class:`newton.solvers.SolverVBD`, which uses an
            AVBD backend for rigid bodies. For cable joints, the stretch and bend behavior
            is defined by the parent/child attachment transforms; the joint axis stored in
            :class:`JointDofConfig` is not currently used directly.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform from the parent body frame to the joint parent anchor frame; its
                translation is the attachment point.
            child_xform (Transform): The transform from the child body frame to the joint child anchor frame; its
                translation is the attachment point.
            stretch_stiffness: Cable stretch stiffness (stored as ``target_ke``) [N/m]. If None, defaults to 1.0e9.
            stretch_damping: Cable stretch damping (stored as ``target_kd``). In :class:`newton.solvers.SolverVBD`
                this is a dimensionless (Rayleigh-style) coefficient. If None,
                defaults to 0.0.
            bend_stiffness: Cable bend/twist stiffness (stored as ``target_ke``) [N*m] (torque per radian). If None,
                defaults to 0.0.
            bend_damping: Cable bend/twist damping (stored as ``target_kd``). In :class:`newton.solvers.SolverVBD`
                this is a dimensionless (Rayleigh-style) coefficient. If None,
                defaults to 0.0.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.
            custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD
                frequency attributes.

        Returns:
            The index of the added joint.

        """
        # Linear DOF (stretch)
        se_ke = 1.0e9 if stretch_stiffness is None else stretch_stiffness
        se_kd = 0.0 if stretch_damping is None else stretch_damping
        ax_lin = ModelBuilder.JointDofConfig(target_ke=se_ke, target_kd=se_kd)

        # Angular DOF (bend/twist)
        bend_ke = 0.0 if bend_stiffness is None else bend_stiffness
        bend_kd = 0.0 if bend_damping is None else bend_damping
        ax_ang = ModelBuilder.JointDofConfig(target_ke=bend_ke, target_kd=bend_kd)

        return self.add_joint(
            JointType.CABLE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[ax_lin],
            angular_axes=[ax_ang],
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            custom_attributes=custom_attributes,
            **kwargs,
        )

    def add_equality_constraint(
        self,
        constraint_type: Any,
        body1: int = -1,
        body2: int = -1,
        anchor: Vec3 | None = None,
        torquescale: float | None = None,
        relpose: Transform | None = None,
        joint1: int = -1,
        joint2: int = -1,
        polycoef: list[float] | None = None,
        key: str | None = None,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Generic method to add any type of equality constraint to this ModelBuilder.

        Args:
            constraint_type (constant): Type of constraint ('connect', 'weld', 'joint')
            body1 (int): Index of the first body participating in the constraint (-1 for world)
            body2 (int): Index of the second body participating in the constraint (-1 for world)
            anchor (Vec3): Anchor point on body1
            torquescale (float): Scales the angular residual for weld
            relpose (Transform): Relative pose of body2 for weld. If None, the identity transform is used.
            joint1 (int): Index of the first joint for joint coupling
            joint2 (int): Index of the second joint for joint coupling
            polycoef (list[float]): Polynomial coefficients for joint coupling
            key (str): Optional constraint name
            enabled (bool): Whether constraint is active
            custom_attributes (dict): Custom attributes to set on the constraint

        Returns:
            Constraint index
        """

        self.equality_constraint_type.append(constraint_type)
        self.equality_constraint_body1.append(body1)
        self.equality_constraint_body2.append(body2)
        self.equality_constraint_anchor.append(anchor or wp.vec3())
        self.equality_constraint_torquescale.append(torquescale)
        self.equality_constraint_relpose.append(relpose or wp.transform_identity())
        self.equality_constraint_joint1.append(joint1)
        self.equality_constraint_joint2.append(joint2)
        self.equality_constraint_polycoef.append(polycoef or [0.0, 0.0, 0.0, 0.0, 0.0])
        self.equality_constraint_key.append(key)
        self.equality_constraint_enabled.append(enabled)
        self.equality_constraint_world.append(self.current_world)

        constraint_idx = len(self.equality_constraint_type) - 1

        # Process custom attributes
        if custom_attributes:
            self._process_custom_attributes(
                entity_index=constraint_idx,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.EQUALITY_CONSTRAINT,
            )

        return constraint_idx

    def add_equality_constraint_connect(
        self,
        body1: int = -1,
        body2: int = -1,
        anchor: Vec3 | None = None,
        key: str | None = None,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a connect equality constraint to the model.
        This constraint connects two bodies at a point. It effectively defines a ball joint outside the kinematic tree.

        Args:
            body1: Index of the first body participating in the constraint (-1 for world)
            body2: Index of the second body participating in the constraint (-1 for world)
            anchor: Anchor point on body1
            key: Optional constraint name
            enabled: Whether constraint is active
            custom_attributes: Custom attributes to set on the constraint

        Returns:
            Constraint index
        """

        return self.add_equality_constraint(
            constraint_type=EqType.CONNECT,
            body1=body1,
            body2=body2,
            anchor=anchor,
            key=key,
            enabled=enabled,
            custom_attributes=custom_attributes,
        )

    def add_equality_constraint_joint(
        self,
        joint1: int = -1,
        joint2: int = -1,
        polycoef: list[float] | None = None,
        key: str | None = None,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a joint equality constraint to the model.
        Constrains the position or angle of one joint to be a quartic polynomial of another joint. Only scalar joint types (prismatic and revolute) can be used.

        Args:
            joint1: Index of the first joint
            joint2: Index of the second joint
            polycoef: Polynomial coefficients for joint coupling
            key: Optional constraint name
            enabled: Whether constraint is active
            custom_attributes: Custom attributes to set on the constraint

        Returns:
            Constraint index
        """

        return self.add_equality_constraint(
            constraint_type=EqType.JOINT,
            joint1=joint1,
            joint2=joint2,
            polycoef=polycoef,
            key=key,
            enabled=enabled,
            custom_attributes=custom_attributes,
        )

    def add_equality_constraint_weld(
        self,
        body1: int = -1,
        body2: int = -1,
        anchor: Vec3 | None = None,
        torquescale: float | None = None,
        relpose: Transform | None = None,
        key: str | None = None,
        enabled: bool = True,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a weld equality constraint to the model.
        Attaches two bodies to each other, removing all relative degrees of freedom between them (softly).

        Args:
            body1: Index of the first body participating in the constraint (-1 for world)
            body2: Index of the second body participating in the constraint (-1 for world)
            anchor: Coordinates of the weld point relative to body2
            torquescale: Scales the angular residual for weld
            relpose (Transform): Relative pose of body2 relative to body1. If None, the identity transform is used
            key: Optional constraint name
            enabled: Whether constraint is active
            custom_attributes: Custom attributes to set on the constraint

        Returns:
            Constraint index
        """

        return self.add_equality_constraint(
            constraint_type=EqType.WELD,
            body1=body1,
            body2=body2,
            anchor=anchor,
            torquescale=torquescale,
            relpose=relpose,
            custom_attributes=custom_attributes,
            key=key,
            enabled=enabled,
        )

    def add_constraint_mimic(
        self,
        joint0: int,
        joint1: int,
        coef0: float = 0.0,
        coef1: float = 1.0,
        enabled: bool = True,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a mimic constraint to the model.

        A mimic constraint enforces that ``joint0 = coef0 + coef1 * joint1``,
        following URDF mimic joint semantics. Both scalar (prismatic, revolute) and
        multi-DOF joints are supported. For multi-DOF joints, the mimic behavior is
        applied equally to all degrees of freedom.

        Args:
            joint0: Index of the follower joint (the one being constrained)
            joint1: Index of the leader joint (the one being mimicked)
            coef0: Offset added after scaling
            coef1: Scale factor applied to joint1's position/angle
            enabled: Whether constraint is active
            key: Optional constraint name
            custom_attributes: Custom attributes to set on the constraint

        Returns:
            Constraint index
        """
        joint_count = self.joint_count
        if joint0 < 0 or joint0 >= joint_count:
            raise ValueError(f"Invalid follower joint index {joint0}; expected 0..{joint_count - 1}")
        if joint1 < 0 or joint1 >= joint_count:
            raise ValueError(f"Invalid leader joint index {joint1}; expected 0..{joint_count - 1}")
        if self.joint_world[joint0] != self.current_world or self.joint_world[joint1] != self.current_world:
            raise ValueError(
                "Mimic constraint joints must belong to the current world. "
                f"joint0_world={self.joint_world[joint0]}, joint1_world={self.joint_world[joint1]}, "
                f"current_world={self.current_world}."
            )

        self.constraint_mimic_joint0.append(joint0)
        self.constraint_mimic_joint1.append(joint1)
        self.constraint_mimic_coef0.append(coef0)
        self.constraint_mimic_coef1.append(coef1)
        self.constraint_mimic_enabled.append(enabled)
        self.constraint_mimic_key.append(key)
        self.constraint_mimic_world.append(self.current_world)

        constraint_idx = len(self.constraint_mimic_joint0) - 1

        # Process custom attributes
        if custom_attributes:
            self._process_custom_attributes(
                entity_index=constraint_idx,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.CONSTRAINT_MIMIC,
            )

        return constraint_idx

    # endregion

    def plot_articulation(
        self,
        show_body_keys=True,
        show_joint_keys=True,
        show_joint_types=True,
        plot_shapes=True,
        show_shape_keys=True,
        show_shape_types=True,
        show_legend=True,
    ):
        """
        Visualizes the model's articulation graph using matplotlib and networkx.
        Uses the spring layout algorithm from networkx to arrange the nodes.
        Bodies are shown as orange squares, shapes are shown as blue circles.

        Args:
            show_body_keys (bool): Whether to show the body keys or indices
            show_joint_keys (bool): Whether to show the joint keys or indices
            show_joint_types (bool): Whether to show the joint types
            plot_shapes (bool): Whether to render the shapes connected to the rigid bodies
            show_shape_keys (bool): Whether to show the shape keys or indices
            show_shape_types (bool): Whether to show the shape geometry types
            show_legend (bool): Whether to show a legend
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        def joint_type_str(type):
            if type == JointType.FREE:
                return "free"
            elif type == JointType.BALL:
                return "ball"
            elif type == JointType.PRISMATIC:
                return "prismatic"
            elif type == JointType.REVOLUTE:
                return "revolute"
            elif type == JointType.D6:
                return "D6"
            elif type == JointType.FIXED:
                return "fixed"
            elif type == JointType.DISTANCE:
                return "distance"
            elif type == JointType.CABLE:
                return "cable"
            return "unknown"

        def shape_type_str(type):
            if type == GeoType.SPHERE:
                return "sphere"
            if type == GeoType.BOX:
                return "box"
            if type == GeoType.CAPSULE:
                return "capsule"
            if type == GeoType.CYLINDER:
                return "cylinder"
            if type == GeoType.CONE:
                return "cone"
            if type == GeoType.MESH:
                return "mesh"
            if type == GeoType.SDF:
                return "sdf"
            if type == GeoType.PLANE:
                return "plane"
            if type == GeoType.CONVEX_MESH:
                return "convex_hull"
            if type == GeoType.NONE:
                return "none"
            return "unknown"

        if show_body_keys:
            vertices = ["world", *self.body_key]
        else:
            vertices = ["-1"] + [str(i) for i in range(self.body_count)]
        if plot_shapes:
            for i in range(self.shape_count):
                shape_label = []
                if show_shape_keys:
                    shape_label.append(self.shape_key[i])
                if show_shape_types:
                    shape_label.append(f"({shape_type_str(self.shape_type[i])})")
                vertices.append("\n".join(shape_label))
        edges = []
        edge_labels = []
        edge_colors = []
        for i in range(self.joint_count):
            edge = (self.joint_child[i] + 1, self.joint_parent[i] + 1)
            edges.append(edge)
            if show_joint_keys:
                joint_label = self.joint_key[i]
            else:
                joint_label = str(i)
            if show_joint_types:
                joint_label += f"\n({joint_type_str(self.joint_type[i])})"
            edge_labels.append(joint_label)
            art_id = self.joint_articulation[i]
            if art_id == -1:
                edge_colors.append("r")
            else:
                edge_colors.append("k")

        if plot_shapes:
            for i in range(self.shape_count):
                edges.append((len(self.body_key) + i + 1, self.shape_body[i] + 1))

        # plot graph
        G = nx.DiGraph()
        for i in range(len(vertices)):
            G.add_node(i, label=vertices[i])
        for i in range(len(edges)):
            label = edge_labels[i] if i < len(edge_labels) else ""
            G.add_edge(edges[i][0], edges[i][1], label=label)
        pos = nx.spring_layout(G, iterations=250)
        # pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_edges(G, pos, node_size=100, edgelist=edges, edge_color=edge_colors, arrows=True)
        # render body vertices
        draw_args = {"node_size": 100}
        bodies = nx.subgraph(G, list(range(self.body_count + 1)))
        nx.draw_networkx_nodes(bodies, pos, node_color="orange", node_shape="s", **draw_args)
        if plot_shapes:
            # render shape vertices
            shapes = nx.subgraph(G, list(range(self.body_count + 1, len(vertices))))
            nx.draw_networkx_nodes(shapes, pos, node_color="skyblue", **draw_args)
            nx.draw_networkx_edges(
                G, pos, node_size=0, edgelist=edges[self.joint_count :], edge_color="gray", style="dashed"
            )
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=6, bbox={"alpha": 0.6, "color": "w", "lw": 0}
        )
        # add node labels
        nx.draw_networkx_labels(G, pos, dict(enumerate(vertices)), font_size=6)
        if show_legend:
            plt.plot([], [], "s", color="orange", label="body")
            plt.plot([], [], "k->", label="joint (child -> parent)")
            if plot_shapes:
                plt.plot([], [], "o", color="skyblue", label="shape")
                plt.plot([], [], "k--", label="shape-body connection")
            plt.legend(loc="upper left", fontsize=6)
        plt.show()

    def collapse_fixed_joints(self, verbose=wp.config.verbose):
        """Removes fixed joints from the model and merges the bodies they connect. This is useful for simplifying the model for faster and more stable simulation."""

        body_data = {}
        body_children = {-1: []}
        visited = {}
        merged_body_data = {}
        for i in range(self.body_count):
            key = self.body_key[i]
            inertia_i = self._coerce_mat33(self.body_inertia[i])
            body_data[i] = {
                "shapes": self.body_shapes[i],
                "q": self.body_q[i],
                "qd": self.body_qd[i],
                "mass": self.body_mass[i],
                "inertia": inertia_i,
                "inv_mass": self.body_inv_mass[i],
                "inv_inertia": self.body_inv_inertia[i],
                "com": wp.vec3(*self.body_com[i]),
                "lock_inertia": self.body_lock_inertia[i],
                "key": key,
                "original_id": i,
            }
            visited[i] = False
            body_children[i] = []

        joint_data = {}
        for i in range(self.joint_count):
            key = self.joint_key[i]
            parent = self.joint_parent[i]
            child = self.joint_child[i]
            body_children[parent].append(child)

            q_start = self.joint_q_start[i]
            qd_start = self.joint_qd_start[i]
            cts_start = self.joint_cts_start[i]
            if i < self.joint_count - 1:
                q_dim = self.joint_q_start[i + 1] - q_start
                qd_dim = self.joint_qd_start[i + 1] - qd_start
                cts_dim = self.joint_cts_start[i + 1] - cts_start
            else:
                q_dim = len(self.joint_q) - q_start
                qd_dim = len(self.joint_qd) - qd_start
                cts_dim = len(self.joint_cts) - cts_start

            data = {
                "type": self.joint_type[i],
                "q": self.joint_q[q_start : q_start + q_dim],
                "qd": self.joint_qd[qd_start : qd_start + qd_dim],
                "cts": self.joint_cts[cts_start : cts_start + cts_dim],
                "armature": self.joint_armature[qd_start : qd_start + qd_dim],
                "q_start": q_start,
                "qd_start": qd_start,
                "cts_start": cts_start,
                "key": key,
                "parent_xform": wp.transform_expand(self.joint_X_p[i]),
                "child_xform": wp.transform_expand(self.joint_X_c[i]),
                "enabled": self.joint_enabled[i],
                "axes": [],
                "axis_dim": self.joint_dof_dim[i],
                "parent": parent,
                "child": child,
                "original_id": i,
            }
            num_lin_axes, num_ang_axes = self.joint_dof_dim[i]
            for j in range(qd_start, qd_start + num_lin_axes + num_ang_axes):
                data["axes"].append(
                    {
                        "axis": self.joint_axis[j],
                        "actuator_mode": self.joint_act_mode[j],
                        "target_ke": self.joint_target_ke[j],
                        "target_kd": self.joint_target_kd[j],
                        "limit_ke": self.joint_limit_ke[j],
                        "limit_kd": self.joint_limit_kd[j],
                        "limit_lower": self.joint_limit_lower[j],
                        "limit_upper": self.joint_limit_upper[j],
                        "target_pos": self.joint_target_pos[j],
                        "target_vel": self.joint_target_vel[j],
                        "effort_limit": self.joint_effort_limit[j],
                    }
                )

            joint_data[(parent, child)] = data

        # sort body children so we traverse the tree in the same order as the bodies are listed
        for children in body_children.values():
            children.sort(key=lambda x: body_data[x]["original_id"])

        # Find bodies referenced in equality constraints that shouldn't be merged into world
        bodies_in_constraints = set()
        for i in range(len(self.equality_constraint_body1)):
            body1 = self.equality_constraint_body1[i]
            body2 = self.equality_constraint_body2[i]
            if body1 >= 0:
                bodies_in_constraints.add(body1)
            if body2 >= 0:
                bodies_in_constraints.add(body2)

        retained_joints = []
        retained_bodies = []
        body_remap = {-1: -1}
        body_merged_parent = {}
        body_merged_transform = {}

        # depth first search over the joint graph
        def dfs(parent_body: int, child_body: int, incoming_xform: wp.transform, last_dynamic_body: int):
            nonlocal visited
            nonlocal retained_joints
            nonlocal retained_bodies
            nonlocal body_data

            joint = joint_data[(parent_body, child_body)]
            # Don't merge fixed joints if the child body is referenced in an equality constraint
            # and would be merged into world (last_dynamic_body == -1)
            should_skip_merge = child_body in bodies_in_constraints and last_dynamic_body == -1

            if should_skip_merge and joint["type"] == JointType.FIXED:
                # Skip merging this fixed joint because the body is referenced in an equality constraint
                if verbose:
                    parent_key = self.body_key[parent_body] if parent_body > -1 else "world"
                    child_key = self.body_key[child_body]
                    print(
                        f"Skipping collapse of fixed joint {joint['key']} between {parent_key} and {child_key}: "
                        f"{child_key} is referenced in an equality constraint and cannot be merged into world"
                    )

            if joint["type"] == JointType.FIXED and not should_skip_merge:
                joint_xform = joint["parent_xform"] * wp.transform_inverse(joint["child_xform"])
                incoming_xform = incoming_xform * joint_xform
                parent_key = self.body_key[parent_body] if parent_body > -1 else "world"
                child_key = self.body_key[child_body]
                last_dynamic_body_key = self.body_key[last_dynamic_body] if last_dynamic_body > -1 else "world"
                if verbose:
                    print(
                        f"Remove fixed joint {joint['key']} between {parent_key} and {child_key}, "
                        f"merging {child_key} into {last_dynamic_body_key}"
                    )
                child_id = body_data[child_body]["original_id"]
                relative_xform = incoming_xform
                merged_body_data[self.body_key[child_body]] = {
                    "relative_xform": relative_xform,
                    "parent_body": self.body_key[parent_body],
                }
                body_merged_parent[child_body] = last_dynamic_body
                body_merged_transform[child_body] = incoming_xform
                for shape in self.body_shapes[child_id]:
                    self.shape_transform[shape] = incoming_xform * self.shape_transform[shape]
                    if verbose:
                        print(
                            f"  Shape {shape} moved to body {last_dynamic_body_key} with transform {self.shape_transform[shape]}"
                        )
                    if last_dynamic_body > -1:
                        self.shape_body[shape] = body_data[last_dynamic_body]["id"]
                        body_data[last_dynamic_body]["shapes"].append(shape)
                    else:
                        self.shape_body[shape] = -1
                        self.body_shapes[-1].append(shape)

                if last_dynamic_body > -1:
                    source_m = body_data[last_dynamic_body]["mass"]
                    source_com = body_data[last_dynamic_body]["com"]
                    # add inertia to last_dynamic_body
                    m = body_data[child_body]["mass"]
                    com = wp.transform_point(incoming_xform, body_data[child_body]["com"])
                    inertia = body_data[child_body]["inertia"]
                    body_data[last_dynamic_body]["inertia"] += transform_inertia(
                        m, inertia, incoming_xform.p, incoming_xform.q
                    )
                    body_data[last_dynamic_body]["mass"] += m
                    body_data[last_dynamic_body]["com"] = (m * com + source_m * source_com) / (m + source_m)
                    # indicate to recompute inverse mass, inertia for this body
                    body_data[last_dynamic_body]["inv_mass"] = None
            else:
                joint["parent_xform"] = incoming_xform * joint["parent_xform"]
                joint["parent"] = last_dynamic_body
                last_dynamic_body = child_body
                incoming_xform = wp.transform()
                retained_joints.append(joint)
                new_id = len(retained_bodies)
                body_data[child_body]["id"] = new_id
                retained_bodies.append(child_body)
                for shape in body_data[child_body]["shapes"]:
                    self.shape_body[shape] = new_id

            visited[parent_body] = True
            if visited[child_body] or child_body not in body_children:
                return
            for child in body_children[child_body]:
                if not visited[child]:
                    dfs(child_body, child, incoming_xform, last_dynamic_body)

        for body in body_children[-1]:
            if not visited[body]:
                dfs(-1, body, wp.transform(), -1)

        # repopulate the model
        # save original body groups before clearing
        original_body_group = self.body_world[:] if self.body_world else []

        self.body_key.clear()
        self.body_q.clear()
        self.body_qd.clear()
        self.body_mass.clear()
        self.body_inertia.clear()
        self.body_com.clear()
        self.body_lock_inertia.clear()
        self.body_inv_mass.clear()
        self.body_inv_inertia.clear()
        self.body_world.clear()  # Clear body groups
        static_shapes = self.body_shapes[-1]
        self.body_shapes.clear()
        # restore static shapes
        self.body_shapes[-1] = static_shapes
        for i in retained_bodies:
            body = body_data[i]
            new_id = len(self.body_key)
            body_remap[body["original_id"]] = new_id
            self.body_key.append(body["key"])
            self.body_q.append(body["q"])
            self.body_qd.append(body["qd"])
            m = body["mass"]
            inertia = body["inertia"]
            self.body_mass.append(m)
            self.body_inertia.append(inertia)
            self.body_com.append(body["com"])
            self.body_lock_inertia.append(body["lock_inertia"])
            if body["inv_mass"] is None:
                # recompute inverse mass and inertia
                if m > 0.0:
                    self.body_inv_mass.append(1.0 / m)
                    self.body_inv_inertia.append(wp.inverse(inertia))
                else:
                    self.body_inv_mass.append(0.0)
                    self.body_inv_inertia.append(wp.mat33(0.0))
            else:
                self.body_inv_mass.append(body["inv_mass"])
                self.body_inv_inertia.append(body["inv_inertia"])
            self.body_shapes[new_id] = body["shapes"]
            # Rebuild body group - use original group if it exists
            if original_body_group and body["original_id"] < len(original_body_group):
                self.body_world.append(original_body_group[body["original_id"]])
            else:
                # If no group was assigned, use default -1
                self.body_world.append(-1)

        # sort joints so they appear in the same order as before
        retained_joints.sort(key=lambda x: x["original_id"])

        joint_remap = {}
        for i, joint in enumerate(retained_joints):
            joint_remap[joint["original_id"]] = i
        # update articulation_start
        for i, old_i in enumerate(self.articulation_start):
            start_i = old_i
            while start_i not in joint_remap:
                start_i += 1
                if start_i >= self.joint_count:
                    break
            self.articulation_start[i] = joint_remap.get(start_i, start_i)
        # remove empty articulation starts, i.e. where the start and end are the same
        self.articulation_start = list(set(self.articulation_start))

        # save original joint worlds and articulations before clearing
        original_ = self.joint_world[:] if self.joint_world else []
        original_articulation = self.joint_articulation[:] if self.joint_articulation else []

        self.joint_key.clear()
        self.joint_type.clear()
        self.joint_parent.clear()
        self.joint_child.clear()
        self.joint_q.clear()
        self.joint_qd.clear()
        self.joint_cts.clear()
        self.joint_q_start.clear()
        self.joint_qd_start.clear()
        self.joint_cts_start.clear()
        self.joint_enabled.clear()
        self.joint_armature.clear()
        self.joint_X_p.clear()
        self.joint_X_c.clear()
        self.joint_axis.clear()
        self.joint_act_mode.clear()
        self.joint_target_ke.clear()
        self.joint_target_kd.clear()
        self.joint_limit_lower.clear()
        self.joint_limit_upper.clear()
        self.joint_limit_ke.clear()
        self.joint_effort_limit.clear()
        self.joint_limit_kd.clear()
        self.joint_dof_dim.clear()
        self.joint_target_pos.clear()
        self.joint_target_vel.clear()
        self.joint_world.clear()
        self.joint_articulation.clear()
        for joint in retained_joints:
            self.joint_key.append(joint["key"])
            self.joint_type.append(joint["type"])
            self.joint_parent.append(body_remap[joint["parent"]])
            self.joint_child.append(body_remap[joint["child"]])
            self.joint_q_start.append(len(self.joint_q))
            self.joint_qd_start.append(len(self.joint_qd))
            self.joint_cts_start.append(len(self.joint_cts))
            self.joint_q.extend(joint["q"])
            self.joint_qd.extend(joint["qd"])
            self.joint_cts.extend(joint["cts"])
            self.joint_armature.extend(joint["armature"])
            self.joint_enabled.append(joint["enabled"])
            self.joint_X_p.append(joint["parent_xform"])
            self.joint_X_c.append(joint["child_xform"])
            self.joint_dof_dim.append(joint["axis_dim"])
            # Rebuild joint world - use original world if it exists
            if original_ and joint["original_id"] < len(original_):
                self.joint_world.append(original_[joint["original_id"]])
            else:
                # If no world was assigned, use default -1
                self.joint_world.append(-1)
            # Rebuild joint articulation assignment
            if original_articulation and joint["original_id"] < len(original_articulation):
                self.joint_articulation.append(original_articulation[joint["original_id"]])
            else:
                self.joint_articulation.append(-1)
            for axis in joint["axes"]:
                self.joint_axis.append(axis["axis"])
                self.joint_act_mode.append(axis["actuator_mode"])
                self.joint_target_ke.append(axis["target_ke"])
                self.joint_target_kd.append(axis["target_kd"])
                self.joint_limit_lower.append(axis["limit_lower"])
                self.joint_limit_upper.append(axis["limit_upper"])
                self.joint_limit_ke.append(axis["limit_ke"])
                self.joint_limit_kd.append(axis["limit_kd"])
                self.joint_target_pos.append(axis["target_pos"])
                self.joint_target_vel.append(axis["target_vel"])
                self.joint_effort_limit.append(axis["effort_limit"])

        # Reset the constraint count based on the retained joints
        self.joint_constraint_count = len(self.joint_cts)

        # Remap equality constraint body/joint indices and transform anchors for merged bodies
        for i in range(len(self.equality_constraint_body1)):
            old_body1 = self.equality_constraint_body1[i]
            old_body2 = self.equality_constraint_body2[i]
            body1_was_merged = False
            body2_was_merged = False

            if old_body1 in body_remap:
                self.equality_constraint_body1[i] = body_remap[old_body1]
            elif old_body1 in body_merged_parent:
                self.equality_constraint_body1[i] = body_remap[body_merged_parent[old_body1]]
                body1_was_merged = True

            if old_body2 in body_remap:
                self.equality_constraint_body2[i] = body_remap[old_body2]
            elif old_body2 in body_merged_parent:
                self.equality_constraint_body2[i] = body_remap[body_merged_parent[old_body2]]
                body2_was_merged = True

            constraint_type = self.equality_constraint_type[i]

            # Transform anchor/relpose from merged body's frame to parent body's frame
            if body1_was_merged:
                merge_xform = body_merged_transform[old_body1]
                if constraint_type == EqType.CONNECT:
                    self.equality_constraint_anchor[i] = wp.transform_point(
                        merge_xform, self.equality_constraint_anchor[i]
                    )
                if constraint_type == EqType.WELD:
                    self.equality_constraint_relpose[i] = merge_xform * self.equality_constraint_relpose[i]

            if body2_was_merged and constraint_type == EqType.WELD:
                merge_xform = body_merged_transform[old_body2]
                self.equality_constraint_anchor[i] = wp.transform_point(merge_xform, self.equality_constraint_anchor[i])
                self.equality_constraint_relpose[i] = self.equality_constraint_relpose[i] * wp.transform_inverse(
                    merge_xform
                )

            old_joint1 = self.equality_constraint_joint1[i]
            old_joint2 = self.equality_constraint_joint2[i]

            if old_joint1 in joint_remap:
                self.equality_constraint_joint1[i] = joint_remap[old_joint1]
            elif old_joint1 != -1:
                if verbose:
                    print(f"Warning: Equality constraint references removed joint {old_joint1}, disabling constraint")
                self.equality_constraint_enabled[i] = False

            if old_joint2 in joint_remap:
                self.equality_constraint_joint2[i] = joint_remap[old_joint2]
            elif old_joint2 != -1:
                if verbose:
                    print(f"Warning: Equality constraint references removed joint {old_joint2}, disabling constraint")
                self.equality_constraint_enabled[i] = False

        # Remap mimic constraint joint indices
        for i in range(len(self.constraint_mimic_joint0)):
            old_joint0 = self.constraint_mimic_joint0[i]
            old_joint1 = self.constraint_mimic_joint1[i]

            if old_joint0 in joint_remap:
                self.constraint_mimic_joint0[i] = joint_remap[old_joint0]
            elif old_joint0 != -1:
                if verbose:
                    print(f"Warning: Mimic constraint references removed joint {old_joint0}, disabling constraint")
                self.constraint_mimic_enabled[i] = False

            if old_joint1 in joint_remap:
                self.constraint_mimic_joint1[i] = joint_remap[old_joint1]
            elif old_joint1 != -1:
                if verbose:
                    print(f"Warning: Mimic constraint references removed joint {old_joint1}, disabling constraint")
                self.constraint_mimic_enabled[i] = False

        # Rebuild parent/child lookups
        self.joint_parents.clear()
        self.joint_children.clear()
        for p, c in zip(self.joint_parent, self.joint_child, strict=True):
            if c not in self.joint_parents:
                self.joint_parents[c] = [p]
            else:
                self.joint_parents[c].append(p)

            if p not in self.joint_children:
                self.joint_children[p] = [c]
            elif c not in self.joint_children[p]:
                self.joint_children[p].append(c)

        return {
            "body_remap": body_remap,
            "joint_remap": joint_remap,
            "body_merged_parent": body_merged_parent,
            "body_merged_transform": body_merged_transform,
            # TODO clean up this data
            "merged_body_data": merged_body_data,
        }

    # muscles
    def add_muscle(
        self, bodies: list[int], positions: list[Vec3], f0: float, lm: float, lt: float, lmax: float, pen: float
    ) -> int:
        """Adds a muscle-tendon activation unit.

        Args:
            bodies: A list of body indices for each waypoint
            positions: A list of positions of each waypoint in the body's local frame
            f0: Force scaling
            lm: Muscle length
            lt: Tendon length
            lmax: Maximally efficient muscle length
            pen: Penalty factor

        Returns:
            The index of the muscle in the model

        .. note:: The simulation support for muscles is in progress and not yet fully functional.

        """

        n = len(bodies)

        self.muscle_start.append(len(self.muscle_bodies))
        self.muscle_params.append((f0, lm, lt, lmax, pen))
        self.muscle_activations.append(0.0)

        for i in range(n):
            self.muscle_bodies.append(bodies[i])
            self.muscle_points.append(positions[i])

        # return the index of the muscle
        return len(self.muscle_start) - 1

    # region shapes

    def add_shape(
        self,
        body: int,
        type: int,
        xform: Transform | None = None,
        cfg: ShapeConfig | None = None,
        scale: Vec3 | None = None,
        src: SDF | Mesh | Any | None = None,
        is_static: bool = False,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a generic collision shape to the model.

        This is the base method for adding shapes; prefer using specific helpers like :meth:`add_shape_sphere` where possible.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body (e.g., static world geometry).
            type (int): The geometry type of the shape (e.g., `GeoType.BOX`, `GeoType.SPHERE`).
            xform (Transform | None): The transform of the shape in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            scale (Vec3 | None): The scale of the geometry. The interpretation depends on the shape type. Defaults to `(1.0, 1.0, 1.0)` if `None`.
            src (SDF | Mesh | Any | None): The source geometry data, e.g., a :class:`Mesh` object for `GeoType.MESH` or an :class:`SDF` object for `GeoType.SDF`. Defaults to `None`.
            is_static (bool): If `True`, the shape will have zero mass, and its density property in `cfg` will be effectively ignored for mass calculation. Typically used for fixed, non-movable collision geometry. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated (e.g., "shape_N"). Defaults to `None`.
            custom_attributes: Dictionary of custom attribute names to values.

        Returns:
            int: The index of the newly added shape.
        """
        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        if cfg is None:
            cfg = self.default_shape_cfg
        cfg.validate()
        if scale is None:
            scale = (1.0, 1.0, 1.0)

        # Validate site invariants
        if cfg.is_site:
            shape_key = key or f"shape_{self.shape_count}"

            # Sites must not have collision enabled
            if cfg.has_shape_collision or cfg.has_particle_collision:
                raise ValueError(
                    f"Site shape '{shape_key}' cannot have collision enabled. "
                    f"Sites must be non-colliding reference points. "
                    f"has_shape_collision={cfg.has_shape_collision}, "
                    f"has_particle_collision={cfg.has_particle_collision}"
                )

            # Sites must have zero density (no mass contribution)
            if cfg.density != 0.0:
                raise ValueError(
                    f"Site shape '{shape_key}' must have zero density. "
                    f"Sites do not contribute to body mass. "
                    f"Got density={cfg.density}"
                )

            # Sites must have collision group 0 (no collision filtering)
            if cfg.collision_group != 0:
                raise ValueError(
                    f"Site shape '{shape_key}' must have collision_group=0. "
                    f"Sites do not participate in collision detection. "
                    f"Got collision_group={cfg.collision_group}"
                )

        self.shape_body.append(body)
        shape = self.shape_count
        if cfg.has_shape_collision:
            # no contacts between shapes of the same body
            for same_body_shape in self.body_shapes[body]:
                self.add_shape_collision_filter_pair(same_body_shape, shape)
        self.body_shapes[body].append(shape)
        self.shape_key.append(key or f"shape_{shape}")
        self.shape_transform.append(xform)
        # Get flags and clear HYDROELASTIC for unsupported shape types (PLANE, HFIELD)
        shape_flags = cfg.flags
        if (shape_flags & ShapeFlags.HYDROELASTIC) and (type == GeoType.PLANE or type == GeoType.HFIELD):
            shape_flags &= (
                ~ShapeFlags.HYDROELASTIC
            )  # Falling back to mesh/primitive collisions for plane and hfield shapes
        self.shape_flags.append(shape_flags)
        self.shape_type.append(type)
        self.shape_scale.append((scale[0], scale[1], scale[2]))
        self.shape_source.append(src)
        self.shape_thickness.append(cfg.thickness)
        self.shape_is_solid.append(cfg.is_solid)
        self.shape_material_ke.append(cfg.ke)
        self.shape_material_kd.append(cfg.kd)
        self.shape_material_kf.append(cfg.kf)
        self.shape_material_ka.append(cfg.ka)
        self.shape_material_mu.append(cfg.mu)
        self.shape_material_restitution.append(cfg.restitution)
        self.shape_material_torsional_friction.append(cfg.torsional_friction)
        self.shape_material_rolling_friction.append(cfg.rolling_friction)
        self.shape_material_k_hydro.append(cfg.k_hydro)
        self.shape_contact_margin.append(
            cfg.contact_margin if cfg.contact_margin is not None else self.rigid_contact_margin
        )
        self.shape_collision_group.append(cfg.collision_group)
        self.shape_collision_radius.append(compute_shape_radius(type, scale, src))
        self.shape_world.append(self.current_world)
        self.shape_sdf_narrow_band_range.append(cfg.sdf_narrow_band_range)
        self.shape_sdf_target_voxel_size.append(cfg.sdf_target_voxel_size)
        self.shape_sdf_max_resolution.append(cfg.sdf_max_resolution)

        if cfg.has_shape_collision and cfg.collision_filter_parent and body > -1 and body in self.joint_parents:
            for parent_body in self.joint_parents[body]:
                if parent_body > -1:
                    for parent_shape in self.body_shapes[parent_body]:
                        self.add_shape_collision_filter_pair(parent_shape, shape)

        if cfg.has_shape_collision and cfg.collision_filter_parent and body > -1 and body in self.joint_children:
            for child_body in self.joint_children[body]:
                for child_shape in self.body_shapes[child_body]:
                    self.add_shape_collision_filter_pair(shape, child_shape)

        if not is_static and cfg.density > 0.0 and body >= 0 and not self.body_lock_inertia[body]:
            (m, c, I) = compute_shape_inertia(type, scale, src, cfg.density, cfg.is_solid, cfg.thickness)
            com_body = wp.transform_point(xform, c)
            self._update_body_mass(body, m, I, com_body, xform.q)

        # Process custom attributes
        if custom_attributes:
            self._process_custom_attributes(
                entity_index=shape,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.SHAPE,
            )

        return shape

    def add_shape_plane(
        self,
        plane: Vec4 | None = (0.0, 0.0, 1.0, 0.0),
        xform: Transform | None = None,
        width: float = 10.0,
        length: float = 10.0,
        body: int = -1,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """
        Adds a plane collision shape to the model.

        If `xform` is provided, it directly defines the plane's position and orientation. The plane's collision normal
        is assumed to be along the local Z-axis of this `xform`.
        If `xform` is `None`, it will be derived from the `plane` equation `a*x + b*y + c*z + d = 0`.
        Plane shapes added via this method are always static (massless).

        Args:
            plane (Vec4 | None): The plane equation `(a, b, c, d)`. If `xform` is `None`, this defines the plane.
                The normal is `(a,b,c)` and `d` is the offset. Defaults to `(0.0, 0.0, 1.0, 0.0)` (an XY ground plane at Z=0) if `xform` is also `None`.
            xform (Transform | None): The transform of the plane in the world or parent body's frame. If `None`, transform is derived from `plane`. Defaults to `None`.
            width (float): The visual/collision extent of the plane along its local X-axis. If `0.0`, considered infinite for collision. Defaults to `10.0`.
            length (float): The visual/collision extent of the plane along its local Y-axis. If `0.0`, considered infinite for collision. Defaults to `10.0`.
            body (int): The index of the parent body this shape belongs to. Use -1 for world-static planes. Defaults to `-1`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute values for SHAPE frequency attributes.

        Returns:
            int: The index of the newly added shape.
        """
        if xform is None:
            assert plane is not None, "Either xform or plane must be provided"
            # compute position and rotation from plane equation
            # For plane equation ax + by + cz + d = 0, the closest point to origin is -(d/||n||) * (n/||n||)
            # where n = (a, b, c). Both the normal and d need to be normalized.
            normal = np.array(plane[:3])
            norm = np.linalg.norm(normal)
            normal /= norm
            d_normalized = plane[3] / norm
            pos = -d_normalized * normal
            # compute rotation from local +Z axis to plane normal
            rot = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*normal))
            xform = wp.transform(pos, rot)
        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(width, length, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.PLANE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            is_static=True,
            key=key,
            custom_attributes=custom_attributes,
        )

    def add_ground_plane(
        self,
        height: float = 0.0,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a ground plane collision shape to the model.

        Args:
            height (float): The vertical offset of the ground plane along the up-vector axis. Positive values raise the plane, negative values lower it. Defaults to `0.0`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        return self.add_shape_plane(
            plane=(*self.up_vector, -height),
            width=0.0,
            length=0.0,
            cfg=cfg,
            key=key or "ground_plane",
        )

    def add_shape_sphere(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        cfg: ShapeConfig | None = None,
        as_site: bool = False,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a sphere collision shape or site to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the sphere in the parent body's local frame. The sphere is centered at this transform's position. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the sphere. Defaults to `1.0`.
            cfg (ShapeConfig | None): The configuration for the shape's properties. If `None`, uses :attr:`default_shape_cfg` (or :attr:`default_site_cfg` when `as_site=True`). If `as_site=True` and `cfg` is provided, a copy is made and site invariants are enforced via `mark_as_site()`. Defaults to `None`.
            as_site (bool): If `True`, creates a site (non-colliding reference point) instead of a collision shape. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute names to values.

        Returns:
            int: The index of the newly added shape or site.
        """
        if cfg is None:
            cfg = self.default_site_cfg if as_site else self.default_shape_cfg
        elif as_site:
            cfg = cfg.copy()
            cfg.mark_as_site()

        scale: Any = wp.vec3(radius, 0.0, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.SPHERE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
            custom_attributes=custom_attributes,
        )

    def add_shape_ellipsoid(
        self,
        body: int,
        xform: Transform | None = None,
        a: float = 1.0,
        b: float = 0.75,
        c: float = 0.5,
        cfg: ShapeConfig | None = None,
        as_site: bool = False,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds an ellipsoid collision shape or site to a body.

        The ellipsoid is centered at its local origin as defined by `xform`, with semi-axes
        `a`, `b`, `c` along the local X, Y, Z axes respectively.

        Note:
            Ellipsoid collision is handled by the GJK/MPR collision pipeline,
            which provides accurate collision detection for all convex shape pairs.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the ellipsoid in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            a (float): The semi-axis of the ellipsoid along its local X-axis. Defaults to `1.0`.
            b (float): The semi-axis of the ellipsoid along its local Y-axis. Defaults to `0.75`.
            c (float): The semi-axis of the ellipsoid along its local Z-axis. Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's properties. If `None`, uses :attr:`default_shape_cfg` (or :attr:`default_site_cfg` when `as_site=True`). If `as_site=True` and `cfg` is provided, a copy is made and site invariants are enforced via `mark_as_site()`. Defaults to `None`.
            as_site (bool): If `True`, creates a site (non-colliding reference point) instead of a collision shape. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute names to values.

        Returns:
            int: The index of the newly added shape or site.

        Example:
            Create an ellipsoid with different semi-axes:

            .. doctest::

                builder = newton.ModelBuilder()
                body = builder.add_body()

                # Add an ellipsoid with semi-axes 1.0, 0.5, 0.25
                builder.add_shape_ellipsoid(
                    body=body,
                    a=1.0,  # X semi-axis
                    b=0.5,  # Y semi-axis
                    c=0.25,  # Z semi-axis
                )

                # A sphere is a special case where a = b = c
                builder.add_shape_ellipsoid(body=body, a=0.5, b=0.5, c=0.5)
        """
        if cfg is None:
            cfg = self.default_site_cfg if as_site else self.default_shape_cfg
        elif as_site:
            cfg = cfg.copy()
            cfg.mark_as_site()

        scale = wp.vec3(a, b, c)
        return self.add_shape(
            body=body,
            type=GeoType.ELLIPSOID,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
            custom_attributes=custom_attributes,
        )

    def add_shape_box(
        self,
        body: int,
        xform: Transform | None = None,
        hx: float = 0.5,
        hy: float = 0.5,
        hz: float = 0.5,
        cfg: ShapeConfig | None = None,
        as_site: bool = False,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a box collision shape or site to a body.

        The box is centered at its local origin as defined by `xform`.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the box in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            hx (float): The half-extent of the box along its local X-axis. Defaults to `0.5`.
            hy (float): The half-extent of the box along its local Y-axis. Defaults to `0.5`.
            hz (float): The half-extent of the box along its local Z-axis. Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's properties. If `None`, uses :attr:`default_shape_cfg` (or :attr:`default_site_cfg` when `as_site=True`). If `as_site=True` and `cfg` is provided, a copy is made and site invariants are enforced via `mark_as_site()`. Defaults to `None`.
            as_site (bool): If `True`, creates a site (non-colliding reference point) instead of a collision shape. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute names to values.

        Returns:
            int: The index of the newly added shape or site.
        """
        if cfg is None:
            cfg = self.default_site_cfg if as_site else self.default_shape_cfg
        elif as_site:
            cfg = cfg.copy()
            cfg.mark_as_site()

        scale = wp.vec3(hx, hy, hz)
        return self.add_shape(
            body=body, type=GeoType.BOX, xform=xform, cfg=cfg, scale=scale, key=key, custom_attributes=custom_attributes
        )

    def add_shape_capsule(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        cfg: ShapeConfig | None = None,
        as_site: bool = False,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a capsule collision shape or site to a body.

        The capsule is centered at its local origin as defined by `xform`. Its length extends along the Z-axis.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the capsule in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the capsule's hemispherical ends and its cylindrical segment. Defaults to `1.0`.
            half_height (float): The half-length of the capsule's central cylindrical segment (excluding the hemispherical ends). Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's properties. If `None`, uses :attr:`default_shape_cfg` (or :attr:`default_site_cfg` when `as_site=True`). If `as_site=True` and `cfg` is provided, a copy is made and site invariants are enforced via `mark_as_site()`. Defaults to `None`.
            as_site (bool): If `True`, creates a site (non-colliding reference point) instead of a collision shape. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute names to values.

        Returns:
            int: The index of the newly added shape or site.
        """
        if cfg is None:
            cfg = self.default_site_cfg if as_site else self.default_shape_cfg
        elif as_site:
            cfg = cfg.copy()
            cfg.mark_as_site()

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)

        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.CAPSULE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
            custom_attributes=custom_attributes,
        )

    def add_shape_cylinder(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        cfg: ShapeConfig | None = None,
        as_site: bool = False,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a cylinder collision shape or site to a body.

        The cylinder is centered at its local origin as defined by `xform`. Its length extends along the Z-axis.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the cylinder in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the cylinder. Defaults to `1.0`.
            half_height (float): The half-length of the cylinder along the Z-axis. Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's properties. If `None`, uses :attr:`default_shape_cfg` (or :attr:`default_site_cfg` when `as_site=True`). If `as_site=True` and `cfg` is provided, a copy is made and site invariants are enforced via `mark_as_site()`. Defaults to `None`.
            as_site (bool): If `True`, creates a site (non-colliding reference point) instead of a collision shape. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute values for SHAPE frequency attributes.

        Returns:
            int: The index of the newly added shape or site.
        """
        if cfg is None:
            cfg = self.default_site_cfg if as_site else self.default_shape_cfg
        elif as_site:
            cfg = cfg.copy()
            cfg.mark_as_site()

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)

        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.CYLINDER,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
            custom_attributes=custom_attributes,
        )

    def add_shape_cone(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        cfg: ShapeConfig | None = None,
        as_site: bool = False,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a cone collision shape to a body.

        The cone's origin is at its geometric center, with the base at -half_height and apex at +half_height along the Z-axis.
        The center of mass is located at -half_height/2 from the origin (1/4 of the total height from the base toward the apex).

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the cone in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the cone's base. Defaults to `1.0`.
            half_height (float): The half-height of the cone (distance from the geometric center to either the base or apex). The total height is 2*half_height. Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            as_site (bool): If `True`, creates a site (non-colliding reference point) instead of a collision shape. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute values for SHAPE frequency attributes.

        Returns:
            int: The index of the newly added shape.
        """
        if cfg is None:
            cfg = self.default_site_cfg if as_site else self.default_shape_cfg
        elif as_site:
            cfg = cfg.copy()
            cfg.mark_as_site()

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)

        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.CONE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
            custom_attributes=custom_attributes,
        )

    def add_shape_mesh(
        self,
        body: int,
        xform: Transform | None = None,
        mesh: Mesh | None = None,
        scale: Vec3 | None = None,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a triangle mesh collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the mesh in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            mesh (Mesh | None): The :class:`Mesh` object containing the vertex and triangle data. Defaults to `None`.
            scale (Vec3 | None): The scale of the mesh. Defaults to `None`, in which case the scale is `(1.0, 1.0, 1.0)`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute values for SHAPE frequency attributes.

        Returns:
            int: The index of the newly added shape.
        """

        if cfg is None:
            cfg = self.default_shape_cfg
        return self.add_shape(
            body=body,
            type=GeoType.MESH,
            xform=xform,
            cfg=cfg,
            scale=scale,
            src=mesh,
            key=key,
            custom_attributes=custom_attributes,
        )

    def add_shape_sdf(
        self,
        body: int,
        xform: Transform | None = None,
        sdf: SDF | None = None,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a signed distance field (SDF) collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the SDF in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            sdf (SDF | None): The :class:`SDF` object representing the signed distance field. Defaults to `None`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.
            custom_attributes: Dictionary of custom attribute values for SHAPE frequency attributes.

        Returns:
            int: The index of the newly added shape.
        """
        if cfg is None:
            cfg = self.default_shape_cfg
        return self.add_shape(
            body=body,
            type=GeoType.SDF,
            xform=xform,
            cfg=cfg,
            src=sdf,
            key=key,
            custom_attributes=custom_attributes,
        )

    def add_shape_convex_hull(
        self,
        body: int,
        xform: Transform | None = None,
        mesh: Mesh | None = None,
        scale: Vec3 | None = None,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a convex hull collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the convex hull in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            mesh (Mesh | None): The :class:`Mesh` object containing the vertex data for the convex hull. Defaults to `None`.
            scale (Vec3 | None): The scale of the convex hull. Defaults to `None`, in which case the scale is `(1.0, 1.0, 1.0)`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if cfg is None:
            cfg = self.default_shape_cfg
        return self.add_shape(
            body=body,
            type=GeoType.CONVEX_MESH,
            xform=xform,
            cfg=cfg,
            scale=scale,
            src=mesh,
            key=key,
        )

    def add_site(
        self,
        body: int,
        xform: Transform | None = None,
        type: int = GeoType.SPHERE,
        scale: Vec3 = (0.01, 0.01, 0.01),
        key: str | None = None,
        visible: bool = False,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a site (non-colliding reference point) to a body.

        Sites are abstract markers that don't participate in physics simulation or collision detection.
        They are useful for:
        - Sensor attachment points (IMU, camera, etc.)
        - Frame of reference definitions
        - Debugging and visualization markers
        - Spatial tendon attachment points (when exported to MuJoCo)

        Args:
            body (int): The index of the parent body this site belongs to. Use -1 for sites not attached to any specific body (for sites defined a at static world position).
            xform (Transform | None): The transform of the site in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            type (int): The geometry type for visualization (e.g., `GeoType.SPHERE`, `GeoType.BOX`). Defaults to `GeoType.SPHERE`.
            scale (Vec3): The scale/size of the site for visualization. Defaults to `(0.01, 0.01, 0.01)`.
            key (str | None): An optional unique key for identifying the site. If `None`, a default key is automatically generated. Defaults to `None`.
            visible (bool): If True, the site will be visible for debugging. If False (default), the site is hidden.
            custom_attributes: Dictionary of custom attribute names to values.

        Returns:
            int: The index of the newly added site (which is stored as a shape internally).

        Example:
            Add an IMU sensor site to a robot torso::

                body = builder.add_body()
                imu_site = builder.add_site(
                    body,
                    xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
                    key="imu_sensor",
                    visible=True,  # Show for debugging
                )
        """
        # Create config for non-colliding site
        cfg = self.default_site_cfg.copy()
        cfg.is_visible = visible

        return self.add_shape(
            body=body,
            type=type,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
            custom_attributes=custom_attributes,
        )

    def approximate_meshes(
        self,
        method: Literal["coacd", "vhacd", "bounding_sphere", "bounding_box"] | RemeshingMethod = "convex_hull",
        shape_indices: list[int] | None = None,
        raise_on_failure: bool = False,
        keep_visual_shapes: bool = False,
        **remeshing_kwargs: dict[str, Any],
    ) -> set[int]:
        """Approximates the mesh shapes of the model.

        The following methods are supported:

        +------------------------+-------------------------------------------------------------------------------+
        | Method                 | Description                                                                   |
        +========================+===============================================================================+
        | ``"coacd"``            | Convex decomposition using `CoACD <https://github.com/wjakob/coacd>`_         |
        +------------------------+-------------------------------------------------------------------------------+
        | ``"vhacd"``            | Convex decomposition using `V-HACD <https://github.com/trimesh/vhacdx>`_      |
        +------------------------+-------------------------------------------------------------------------------+
        | ``"bounding_sphere"``  | Approximate the mesh with a sphere                                            |
        +------------------------+-------------------------------------------------------------------------------+
        | ``"bounding_box"``     | Approximate the mesh with an oriented bounding box                            |
        +------------------------+-------------------------------------------------------------------------------+
        | ``"convex_hull"``      | Approximate the mesh with a convex hull (default)                             |
        +------------------------+-------------------------------------------------------------------------------+
        | ``<remeshing_method>`` | Any remeshing method supported by :func:`newton.geometry.remesh_mesh`         |
        +------------------------+-------------------------------------------------------------------------------+

        .. note::

            The ``coacd`` and ``vhacd`` methods require additional dependencies (``coacd`` or ``trimesh`` and ``vhacdx`` respectively) to be installed.
            The convex hull approximation requires ``scipy`` to be installed.

        The ``raise_on_failure`` parameter controls the behavior when the remeshing fails:
            - If `True`, an exception is raised when the remeshing fails.
            - If `False`, a warning is logged, and the method falls back to the next available method in the order of preference:
                - If convex decomposition via CoACD or V-HACD fails or dependencies are not available, the method will fall back to using the ``convex_hull`` method.
                - If convex hull approximation fails, it will fall back to the ``bounding_box`` method.

        Args:
            method: The method to use for approximating the mesh shapes.
            shape_indices: The indices of the shapes to simplify. If `None`, all mesh shapes that have the :attr:`ShapeFlags.COLLIDE_SHAPES` flag set are simplified.
            raise_on_failure: If `True`, raises an exception if the remeshing fails. If `False`, it will log a warning and continue with the fallback method.
            **remeshing_kwargs: Additional keyword arguments passed to the remeshing function.

        Returns:
            set[int]: A set of indices of the shapes that were successfully remeshed.
        """
        remeshing_methods = [*RemeshingMethod.__args__, "coacd", "vhacd", "bounding_sphere", "bounding_box"]
        if method not in remeshing_methods:
            raise ValueError(
                f"Unsupported remeshing method: {method}. Supported methods are: {', '.join(remeshing_methods)}."
            )

        if shape_indices is None:
            shape_indices = [
                i
                for i, stype in enumerate(self.shape_type)
                if stype == GeoType.MESH and self.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES
            ]

        if keep_visual_shapes:
            # if keeping visual shapes, first copy input shapes, mark the copies as visual-only,
            # and mark the originals as non-visible.
            # in the rare event that approximation fails, we end up with two identical shapes,
            # one collision-only, one visual-only, but this simplifies the logic below.
            for shape in shape_indices:
                if not (self.shape_flags[shape] & ShapeFlags.VISIBLE):
                    continue

                body = self.shape_body[shape]
                xform = self.shape_transform[shape]
                cfg = ModelBuilder.ShapeConfig(
                    density=0.0,  # do not add extra mass / inertia
                    thickness=self.shape_thickness[shape],
                    is_solid=self.shape_is_solid[shape],
                    has_shape_collision=False,
                    has_particle_collision=False,
                    is_visible=True,
                )
                self.add_shape_mesh(
                    body=body,
                    xform=xform,
                    cfg=cfg,
                    mesh=self.shape_source[shape],
                    key=f"{self.shape_key[shape]}_visual",
                    scale=self.shape_scale[shape],
                )

                # disable visibility of the original shape
                self.shape_flags[shape] &= ~ShapeFlags.VISIBLE

        # keep track of remeshed shapes to handle fallbacks
        remeshed_shapes = set()

        if method == "coacd" or method == "vhacd":
            try:
                if method == "coacd":
                    # convex decomposition using CoACD
                    import coacd
                else:
                    # convex decomposition using V-HACD
                    import trimesh

                decompositions = {}

                for shape in shape_indices:
                    mesh: Mesh = self.shape_source[shape]
                    scale = self.shape_scale[shape]
                    hash_m = hash(mesh)
                    if hash_m in decompositions:
                        decomposition = decompositions[hash_m]
                    else:
                        if method == "coacd":
                            cmesh = coacd.Mesh(mesh.vertices, mesh.indices.reshape(-1, 3))
                            coacd_settings = {
                                "threshold": 0.5,
                                "mcts_nodes": 20,
                                "mcts_iterations": 5,
                                "mcts_max_depth": 1,
                                "merge": False,
                                "max_convex_hull": mesh.maxhullvert,
                            }
                            coacd_settings.update(remeshing_kwargs)
                            decomposition = coacd.run_coacd(cmesh, **coacd_settings)
                        else:
                            tmesh = trimesh.Trimesh(mesh.vertices, mesh.indices.reshape(-1, 3))
                            vhacd_settings = {
                                "maxNumVerticesPerCH": mesh.maxhullvert,
                            }
                            vhacd_settings.update(remeshing_kwargs)
                            decomposition = trimesh.decomposition.convex_decomposition(tmesh, **vhacd_settings)
                            decomposition = [(d["vertices"], d["faces"]) for d in decomposition]
                        decompositions[hash_m] = decomposition
                    if len(decomposition) == 0:
                        continue
                    # note we need to copy the mesh to avoid modifying the original mesh
                    self.shape_source[shape] = self.shape_source[shape].copy(
                        vertices=decomposition[0][0], indices=decomposition[0][1]
                    )
                    # mark as convex mesh type
                    self.shape_type[shape] = GeoType.CONVEX_MESH
                    if len(decomposition) > 1:
                        body = self.shape_body[shape]
                        xform = self.shape_transform[shape]
                        cfg = ModelBuilder.ShapeConfig(
                            density=0.0,  # do not add extra mass / inertia
                            ke=self.shape_material_ke[shape],
                            kd=self.shape_material_kd[shape],
                            kf=self.shape_material_kf[shape],
                            ka=self.shape_material_ka[shape],
                            mu=self.shape_material_mu[shape],
                            restitution=self.shape_material_restitution[shape],
                            torsional_friction=self.shape_material_torsional_friction[shape],
                            rolling_friction=self.shape_material_rolling_friction[shape],
                            thickness=self.shape_thickness[shape],
                            is_solid=self.shape_is_solid[shape],
                            collision_group=self.shape_collision_group[shape],
                            collision_filter_parent=self.default_shape_cfg.collision_filter_parent,
                        )
                        cfg.flags = self.shape_flags[shape]
                        for i in range(1, len(decomposition)):
                            # add additional convex parts as convex meshes
                            self.add_shape_convex_hull(
                                body=body,
                                xform=xform,
                                mesh=Mesh(decomposition[i][0], decomposition[i][1]),
                                scale=scale,
                                cfg=cfg,
                                key=f"{self.shape_key[shape]}_convex_{i}",
                            )
                    remeshed_shapes.add(shape)
            except Exception as e:
                if raise_on_failure:
                    raise RuntimeError(f"Remeshing with method '{method}' failed.") from e
                else:
                    warnings.warn(
                        f"Remeshing with method '{method}' failed: {e}. Falling back to convex_hull.", stacklevel=2
                    )
                    method = "convex_hull"

        if method in RemeshingMethod.__args__:
            # remeshing of the individual meshes
            remeshed = {}
            for shape in shape_indices:
                if shape in remeshed_shapes:
                    # already remeshed with coacd or vhacd
                    continue
                mesh: Mesh = self.shape_source[shape]
                hash_m = hash(mesh)
                rmesh = remeshed.get(hash_m, None)
                if rmesh is None:
                    try:
                        rmesh = remesh_mesh(mesh, method=method, inplace=False, **remeshing_kwargs)
                        remeshed[hash_m] = rmesh
                    except Exception as e:
                        if raise_on_failure:
                            raise RuntimeError(f"Remeshing with method '{method}' failed for shape {shape}.") from e
                        else:
                            warnings.warn(
                                f"Remeshing with method '{method}' failed for shape {shape}: {e}. Falling back to bounding_box.",
                                stacklevel=2,
                            )
                            continue
                # note we need to copy the mesh to avoid modifying the original mesh
                self.shape_source[shape] = self.shape_source[shape].copy(vertices=rmesh.vertices, indices=rmesh.indices)
                # mark convex_hull result as convex mesh type for efficient collision detection
                if method == "convex_hull":
                    self.shape_type[shape] = GeoType.CONVEX_MESH
                remeshed_shapes.add(shape)

        if method == "bounding_box":
            for shape in shape_indices:
                if shape in remeshed_shapes:
                    continue
                mesh: Mesh = self.shape_source[shape]
                scale = self.shape_scale[shape]
                vertices = mesh.vertices * np.array([*scale])
                tf, scale = compute_inertia_obb(vertices)
                self.shape_type[shape] = GeoType.BOX
                self.shape_source[shape] = None
                self.shape_scale[shape] = scale
                shape_tf = wp.transform(*self.shape_transform[shape])
                self.shape_transform[shape] = shape_tf * tf
                remeshed_shapes.add(shape)
        elif method == "bounding_sphere":
            for shape in shape_indices:
                if shape in remeshed_shapes:
                    continue
                mesh: Mesh = self.shape_source[shape]
                scale = self.shape_scale[shape]
                vertices = mesh.vertices * np.array([*scale])
                center = np.mean(vertices, axis=0)
                radius = np.max(np.linalg.norm(vertices - center, axis=1))
                self.shape_type[shape] = GeoType.SPHERE
                self.shape_source[shape] = None
                self.shape_scale[shape] = wp.vec3(radius, 0.0, 0.0)
                tf = wp.transform(center, wp.quat_identity())
                shape_tf = wp.transform(*self.shape_transform[shape])
                self.shape_transform[shape] = shape_tf * tf
                remeshed_shapes.add(shape)

        return remeshed_shapes

    def add_rod(
        self,
        positions: list[Vec3],
        quaternions: list[Quat] | None = None,
        radius: float = 0.1,
        cfg: ShapeConfig | None = None,
        stretch_stiffness: float | None = None,
        stretch_damping: float | None = None,
        bend_stiffness: float | None = None,
        bend_damping: float | None = None,
        closed: bool = False,
        key: str | None = None,
        wrap_in_articulation: bool = True,
    ) -> tuple[list[int], list[int]]:
        """Adds a rod composed of capsule bodies connected by cable joints.

        Constructs a chain of capsule bodies from the given centerline points and orientations.
        Each segment is a capsule aligned by the corresponding quaternion, and adjacent capsules
        are connected by cable joints providing one linear (stretch) and one angular (bend/twist)
        degree of freedom.

        Args:
            positions: Centerline node positions (segment endpoints) in world space. These are the
                tip/end points of the capsules, with one extra point so that for ``N`` segments there
                are ``N+1`` positions.
            quaternions: Optional per-segment (per-edge) orientations in world space. If provided,
                must have ``len(positions) - 1`` elements and each quaternion should align the capsule's
                local +Z with the segment direction ``positions[i+1] - positions[i]``. If None,
                orientations are computed automatically to align +Z with each segment direction.
            radius: Capsule radius.
            cfg: Shape configuration for the capsules. If None, :attr:`default_shape_cfg` is used.
            stretch_stiffness: Stretch stiffness for the cable joints. For rods, this is treated as a
                material-like axial/shear stiffness (commonly interpreted as EA)
                with units [N] and is internally converted to an effective point stiffness [N/m] by dividing by
                segment length. If None, defaults to 1.0e9.
            stretch_damping: Stretch damping for the cable joints (applied per-joint; not length-normalized). If None,
                defaults to 0.0.
            bend_stiffness: Bend/twist stiffness for the cable joints. For rods, this is treated as a
                material-like bending/twist stiffness (e.g., EI) with units [N*m^2] and is internally converted to
                an effective per-joint stiffness [N*m] (torque per radian) by dividing by segment length. If None,
                defaults to 0.0.
            bend_damping: Bend/twist damping for the cable joints (applied per-joint; not length-normalized). If None,
                defaults to 0.0.
            closed: If True, connects the last segment back to the first to form a closed loop. If False,
                creates an open chain. Note: rods require at least 2 segments.
            key: Optional key prefix for bodies, shapes, and joints.
            wrap_in_articulation: If True, the created joints are automatically wrapped into a single
                articulation. Defaults to True to ensure valid simulation models.

        Returns:
            tuple[list[int], list[int]]: (body_indices, joint_indices). For an open chain,
            ``len(joint_indices) == num_segments - 1``; for a closed loop, ``len(joint_indices) == num_segments``.

        Articulations:
            By default (``wrap_in_articulation=True``), the created joints are wrapped into a single
            articulation, which avoids orphan joints during :meth:`finalize`.
            If ``wrap_in_articulation=False``, this method will return the created joint indices but will
            not wrap them; callers must place them into one or more articulations (via :meth:`add_articulation`)
            before calling :meth:`finalize`.

        Raises:
            ValueError: If ``positions`` and ``quaternions`` lengths are incompatible.
            ValueError: If the rod has fewer than 2 segments.

        Note:
            - Bend defaults are 0.0 (no bending resistance unless specified). Stretch defaults to a high
              stiffness (1.0e9), which keeps neighboring capsules closely coupled (approximately inextensible).
            - Internally, stretch and bend stiffnesses are pre-scaled by dividing by segment length so solver kernels
              do not need per-segment length normalization.
            - Damping values are passed through as provided (per joint) and are not length-normalized.
            - Each segment is implemented as a capsule primitive. The segment's body transform is
              placed at the start point ``positions[i]`` with a local center-of-mass offset of
              ``(0, 0, half_height)`` so that the COM lies at the segment midpoint. The capsule shape
              is added with a local transform of ``(0, 0, half_height)`` so it spans from the start to
              the end along local +Z.
        """
        if cfg is None:
            cfg = self.default_shape_cfg

        # Stretch defaults: high stiffness to keep neighboring capsules tightly coupled
        stretch_stiffness = 1.0e9 if stretch_stiffness is None else stretch_stiffness
        stretch_damping = 0.0 if stretch_damping is None else stretch_damping

        # Bend defaults: 0.0 (users must explicitly set for bending resistance)
        bend_stiffness = 0.0 if bend_stiffness is None else bend_stiffness
        bend_damping = 0.0 if bend_damping is None else bend_damping

        # Input validation
        if stretch_stiffness < 0.0 or bend_stiffness < 0.0:
            raise ValueError("add_rod: stretch_stiffness and bend_stiffness must be >= 0")

        num_segments = len(positions) - 1
        if num_segments < 1:
            raise ValueError("add_rod: positions must contain at least 2 points")

        # Coerce all input positions to wp.vec3 so arithmetic (p1 - p0), wp.length, wp.normalize
        # always operate on Warp vector types even if the caller passed tuples/lists.
        positions_wp: list[wp.vec3] = [axis_to_vec3(p) for p in positions]

        if quaternions is not None and len(quaternions) != num_segments:
            raise ValueError(
                f"add_rod: quaternions must have {num_segments} elements for {num_segments} segments, "
                f"got {len(quaternions)} quaternions"
            )

        if num_segments < 2:
            # A "rod" in this API is defined as multiple capsules coupled by cable joints.
            # If you want a single capsule, create a body + capsule shape directly.
            raise ValueError(
                f"add_rod: requires at least 2 segments (got {num_segments}); "
                "for a single capsule, create a body and add a capsule shape instead."
            )

        # Build linear graph edges: (0, 1), (1, 2), ..., (N-1, N)
        # Note: positions has N+1 elements for N segments.
        edges = [(i, i + 1) for i in range(num_segments)]

        # Delegate to add_rod_graph to create bodies and internal joints.
        # We use wrap_in_articulation=False and let add_rod manage articulation wrapping so that:
        # - open chains are wrapped into a single articulation (tree), and
        # - closed loops add one extra "loop joint" after wrapping, which must not be part of an articulation.
        link_bodies, link_joints = self.add_rod_graph(
            node_positions=positions_wp,
            edges=edges,
            radius=radius,
            cfg=cfg,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            key=key,
            wrap_in_articulation=False,
            quaternions=quaternions,
        )

        # Wrap all joints into an articulation if requested.
        if wrap_in_articulation and link_joints:
            rod_art_key = f"{key}_articulation" if key else None
            self.add_articulation(link_joints, key=rod_art_key)

        # For closed loops, add one extra loop-closing cable joint that is intentionally
        # *not* part of an articulation (articulations must be trees/forests).
        if closed:
            if not wrap_in_articulation:
                warnings.warn(
                    "add_rod: wrap_in_articulation=False requires the caller to wrap joints via add_articulation() "
                    "before finalize; closed=True also adds a loop-closing joint that must remain outside any "
                    "articulation.",
                    UserWarning,
                    stacklevel=2,
                )

            if link_bodies:
                first_body = link_bodies[0]
                last_body = link_bodies[-1]

                # Connect the end of the last segment to the start of the first segment.
                L_last = float(wp.length(positions_wp[-1] - positions_wp[-2]))
                min_segment_length = 1.0e-9
                if L_last <= min_segment_length:
                    L_last = min_segment_length

                parent_xform = wp.transform(wp.vec3(0.0, 0.0, L_last), wp.quat_identity())
                child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

                # Normalize stiffness by segment length, consistent with add_rod_graph().
                stretch_ke_eff = stretch_stiffness / L_last
                bend_ke_eff = bend_stiffness / L_last

                loop_joint_key = f"{key}_cable_{len(link_joints) + 1}" if key else None
                j_loop = self.add_joint_cable(
                    parent=last_body,
                    child=first_body,
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    bend_stiffness=bend_ke_eff,
                    bend_damping=bend_damping,
                    stretch_stiffness=stretch_ke_eff,
                    stretch_damping=stretch_damping,
                    key=loop_joint_key,
                    collision_filter_parent=True,
                    enabled=True,
                )
                link_joints.append(j_loop)

        return link_bodies, link_joints

    def add_rod_graph(
        self,
        node_positions: list[Vec3],
        edges: list[tuple[int, int]],
        radius: float = 0.1,
        cfg: ShapeConfig | None = None,
        stretch_stiffness: float | None = None,
        stretch_damping: float | None = None,
        bend_stiffness: float | None = None,
        bend_damping: float | None = None,
        key: str | None = None,
        wrap_in_articulation: bool = True,
        quaternions: list[Quat] | None = None,
        junction_collision_filter: bool = True,
    ) -> tuple[list[int], list[int]]:
        """Adds a rod/cable *graph* (supports junctions) from nodes + edges.

        This is a generalization of :meth:`add_rod` to support branching/junction topologies.

        Representation:

        - Each *edge* becomes a capsule rigid body spanning from ``node_positions[u]`` to
          ``node_positions[v]`` (body frame is placed at the start node ``u`` and local +Z points
          toward ``v``).
        - Cable joints are created between edge-bodies that share a node, using a spanning-tree
          traversal so that each body has a single parent when wrapped into an articulation.

        Notes:

        - If ``wrap_in_articulation=True`` (default), joints are created as a forest (one
          articulation per connected component). This keeps the joint graph articulation-safe
          (tree/forest), avoiding cycles at junctions.
        - Cycles in the edge adjacency graph are *not* explicitly closed with extra joints when
          ``wrap_in_articulation=True`` (cycles would violate articulation tree constraints). If
          you need closed loops, build them explicitly without articulation wrapping.
        - If ``wrap_in_articulation=False``, joints are created directly at each node to connect
          all incident edges. This can preserve rings/loops, but does not produce an articulation
          tree (edges may effectively have multiple "parents" in the joint graph).

        Args:
            node_positions: Junction node positions in world space.
            edges: List of (u, v) node index pairs defining rod segments. Each edge creates one
                capsule body oriented so its local +Z points from node ``u`` to node ``v``.
            radius: Capsule radius.
            cfg: Shape configuration for the capsules. If None, :attr:`default_shape_cfg` is used.
            stretch_stiffness: Material-like axial stiffness (EA) [N], normalized by edge length
                into an effective joint stiffness [N/m]. Defaults to 1.0e9.
            stretch_damping: Stretch damping (per joint). Defaults to 0.0.
            bend_stiffness: Material-like bend/twist stiffness (EI) [N*m^2], normalized by edge
                length into an effective joint stiffness [N*m]. Defaults to 0.0.
            bend_damping: Bend/twist damping (per joint). Defaults to 0.0.
            key: Optional key prefix for bodies, shapes, joints, and articulations.
            wrap_in_articulation: If True, wraps the generated joint forest into one articulation
                per connected component.
            quaternions: Optional per-edge orientations in world space. If provided, must have
                ``len(edges)`` elements and each quaternion must align the capsule's local +Z with
                the corresponding edge direction ``node_positions[v] - node_positions[u]``. If
                None, orientations are computed automatically to align +Z with each edge direction.
            junction_collision_filter: If True, adds collision filters between *non-jointed* segment
                bodies that are incident to a junction node (degree >= 3). This prevents immediate
                self-collision impulses at welded junctions, even though the joint set is a spanning
                tree (so not all incident body pairs are directly jointed).

        Returns:
            tuple[list[int], list[int]]: (body_indices, joint_indices) where bodies correspond to
            edges in the same order as ``edges``.
        """
        if cfg is None:
            cfg = self.default_shape_cfg

        # Stretch defaults: high stiffness to keep neighboring capsules tightly coupled
        stretch_stiffness = 1.0e9 if stretch_stiffness is None else stretch_stiffness
        stretch_damping = 0.0 if stretch_damping is None else stretch_damping

        # Bend defaults: 0.0 (users must explicitly set for bending resistance)
        bend_stiffness = 0.0 if bend_stiffness is None else bend_stiffness
        bend_damping = 0.0 if bend_damping is None else bend_damping

        if stretch_stiffness < 0.0 or bend_stiffness < 0.0:
            raise ValueError("add_rod_graph: stretch_stiffness and bend_stiffness must be >= 0")
        if len(node_positions) < 2:
            raise ValueError("add_rod_graph: node_positions must contain at least 2 nodes")
        if len(edges) < 1:
            raise ValueError("add_rod_graph: edges must contain at least 1 edge")

        num_nodes = len(node_positions)
        num_edges = len(edges)
        if quaternions is not None and len(quaternions) != num_edges:
            raise ValueError(
                f"add_rod_graph: quaternions must have {num_edges} elements for {num_edges} edges, "
                f"got {len(quaternions)} quaternions"
            )

        # Guard against near-zero lengths: edge length is used to normalize stiffness (EA/L, EI/L).
        min_segment_length = 1.0e-9

        # Coerce all input node positions to wp.vec3 so arithmetic (p1 - p0), wp.length, wp.normalize
        # always operate on Warp vector types even if the caller passed tuples/lists.
        node_positions_wp: list[wp.vec3] = [axis_to_vec3(p) for p in node_positions]

        # Build per-node incidence for spanning-tree traversal.
        node_incidence: list[list[int]] = [[] for _ in range(num_nodes)]

        # Per-edge data
        edge_u: list[int] = []
        edge_v: list[int] = []
        edge_len: list[float] = []
        edge_bodies: list[int] = []

        # Create all edge bodies first.
        for e_idx, (u, v) in enumerate(edges):
            if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
                raise ValueError(
                    f"add_rod_graph: edge {e_idx} has invalid node indices ({u}, {v}) for {num_nodes} nodes"
                )
            if u == v:
                raise ValueError(f"add_rod_graph: edge {e_idx} connects a node to itself ({u} -> {v})")

            p0 = node_positions_wp[u]
            p1 = node_positions_wp[v]
            seg_vec = p1 - p0
            seg_length = float(wp.length(seg_vec))
            if seg_length <= min_segment_length:
                raise ValueError(
                    f"add_rod_graph: edge {e_idx} has a too-small length (length={seg_length:.3e}); "
                    f"segment length must be > {min_segment_length:.1e}"
                )

            if quaternions is None:
                seg_dir = wp.normalize(seg_vec)
                q = quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), seg_dir)
            else:
                q = quaternions[e_idx]

                # Local +Z must align with the segment direction.
                seg_dir = wp.normalize(seg_vec)
                local_z_world = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
                alignment = wp.dot(seg_dir, local_z_world)
                if alignment < 0.999:
                    raise ValueError(
                        "add_rod_graph: quaternion at edge index "
                        f"{e_idx} does not align capsule +Z with edge direction (node_positions[v] - node_positions[u]); "
                        "quaternions must be world-space and constructed so that local +Z maps to the "
                        "edge direction node_positions[v] - node_positions[u]."
                    )
            half_height = 0.5 * seg_length

            # Position body at start node, with COM offset to segment center
            body_q = wp.transform(p0, q)
            com_offset = wp.vec3(0.0, 0.0, half_height)

            body_key = f"{key}_edge_body_{e_idx}" if key else None
            shape_key = f"{key}_edge_capsule_{e_idx}" if key else None

            body_id = self.add_link(xform=body_q, com=com_offset, key=body_key)

            # Place capsule so it spans from start to end along +Z
            capsule_xform = wp.transform(wp.vec3(0.0, 0.0, half_height), wp.quat_identity())

            self.add_shape_capsule(
                body_id,
                xform=capsule_xform,
                radius=radius,
                half_height=half_height,
                cfg=cfg,
                key=shape_key,
            )

            edge_u.append(u)
            edge_v.append(v)
            edge_len.append(seg_length)
            edge_bodies.append(body_id)

            node_incidence[u].append(e_idx)
            node_incidence[v].append(e_idx)

        def _edge_anchor_xform(e_idx: int, node_idx: int) -> wp.transform:
            # Body frame is at start node u; end node v is at local z = edge_len.
            if node_idx == edge_u[e_idx]:
                z = 0.0
            elif node_idx == edge_v[e_idx]:
                z = edge_len[e_idx]
            else:
                raise RuntimeError("add_rod_graph: internal error (node not incident to edge)")
            return wp.transform(wp.vec3(0.0, 0.0, float(z)), wp.quat_identity())

        joint_counter = 0
        jointed_body_pairs: set[tuple[int, int]] = set()

        def _remember_jointed_pair(parent_body: int, child_body: int) -> None:
            # Canonical order so lookups are symmetric.
            if parent_body <= child_body:
                jointed_body_pairs.add((parent_body, child_body))
            else:
                jointed_body_pairs.add((child_body, parent_body))

        def _build_joints_star() -> list[int]:
            """Builds joints by connecting incident edges directly at each node."""
            nonlocal joint_counter
            all_joints: list[int] = []

            # No articulation constraints: connect incident edges directly at each node.
            # This preserves cycles (rings/loops) but can create multi-parent relationships, which is
            # fine when not wrapping into an articulation.
            for node_idx in range(num_nodes):
                inc = node_incidence[node_idx]
                if len(inc) < 2:
                    continue

                # Deterministic parent choice: use the first edge in incidence list.
                # Since node_incidence is built by iterating edges in order (0, 1, 2...),
                # this implicitly picks the edge with the lowest index as the parent.
                parent_edge = inc[0]
                parent_body = edge_bodies[parent_edge]
                parent_xform = _edge_anchor_xform(parent_edge, node_idx)

                for child_edge in inc[1:]:
                    child_body = edge_bodies[child_edge]
                    if parent_body == child_body:
                        raise RuntimeError("add_rod_graph: internal error (self-connection)")

                    child_xform = _edge_anchor_xform(child_edge, node_idx)

                    # Normalize stiffness by segment length, consistent with add_rod().
                    # Use a symmetric length so stiffness is traversal/order invariant.
                    L_parent = edge_len[parent_edge]
                    L_child = edge_len[child_edge]
                    L_sym = 0.5 * (L_parent + L_child)
                    stretch_ke_eff = stretch_stiffness / L_sym
                    bend_ke_eff = bend_stiffness / L_sym

                    joint_counter += 1
                    joint_key = f"{key}_cable_{joint_counter}" if key else None

                    j = self.add_joint_cable(
                        parent=parent_body,
                        child=child_body,
                        parent_xform=parent_xform,
                        child_xform=child_xform,
                        bend_stiffness=bend_ke_eff,
                        bend_damping=bend_damping,
                        stretch_stiffness=stretch_ke_eff,
                        stretch_damping=stretch_damping,
                        key=joint_key,
                        collision_filter_parent=True,
                        enabled=True,
                    )
                    all_joints.append(j)
                    _remember_jointed_pair(parent_body, child_body)
            return all_joints

        def _build_joints_forest() -> list[int]:
            """Builds joints using a spanning-forest traversal to ensure articulation-safe (tree) topology."""
            nonlocal joint_counter
            all_joints: list[int] = []
            visited = [False] * num_edges
            component_index = 0

            for start_edge in range(num_edges):
                if visited[start_edge]:
                    continue

                # BFS over edges
                queue: deque[int] = deque([start_edge])
                visited[start_edge] = True
                component_joints: list[int] = []
                component_edges: list[int] = []

                while queue:
                    parent_edge = queue.popleft()
                    component_edges.append(parent_edge)
                    parent_body = edge_bodies[parent_edge]

                    for shared_node in (edge_u[parent_edge], edge_v[parent_edge]):
                        for child_edge in node_incidence[shared_node]:
                            if child_edge == parent_edge or visited[child_edge]:
                                continue

                            child_body = edge_bodies[child_edge]
                            if parent_body == child_body:
                                raise RuntimeError("add_rod_graph: internal error (self-connection)")

                            # Anchors at the shared node on each edge body
                            parent_xform = _edge_anchor_xform(parent_edge, shared_node)
                            child_xform = _edge_anchor_xform(child_edge, shared_node)

                            # Normalize stiffness by segment length, consistent with add_rod().
                            # Use a symmetric length so stiffness is traversal/order invariant.
                            L_parent = edge_len[parent_edge]
                            L_child = edge_len[child_edge]
                            L_sym = 0.5 * (L_parent + L_child)
                            stretch_ke_eff = stretch_stiffness / L_sym
                            bend_ke_eff = bend_stiffness / L_sym

                            joint_counter += 1
                            joint_key = f"{key}_cable_{joint_counter}" if key else None

                            j = self.add_joint_cable(
                                parent=parent_body,
                                child=child_body,
                                parent_xform=parent_xform,
                                child_xform=child_xform,
                                bend_stiffness=bend_ke_eff,
                                bend_damping=bend_damping,
                                stretch_stiffness=stretch_ke_eff,
                                stretch_damping=stretch_damping,
                                key=joint_key,
                                collision_filter_parent=True,
                                enabled=True,
                            )

                            component_joints.append(j)
                            all_joints.append(j)
                            _remember_jointed_pair(parent_body, child_body)
                            visited[child_edge] = True
                            queue.append(child_edge)

                # If the original node-edge graph contains a cycle, we cannot "close" it with extra
                # joints while keeping an articulation tree. Warn so callers don't assume rings/loops
                # are preserved under `wrap_in_articulation=True`.
                if component_edges:
                    component_nodes: set[int] = set()
                    for e_idx in component_edges:
                        component_nodes.add(edge_u[e_idx])
                        component_nodes.add(edge_v[e_idx])

                    # Undirected graph cycle condition: E > V - 1 (for any connected component).
                    if len(component_edges) > max(0, len(component_nodes) - 1):
                        warnings.warn(
                            "add_rod_graph: detected a cycle (closed loop) in the edge graph. "
                            "With wrap_in_articulation=True, joints are built as a tree/forest, so "
                            "cycles are not closed. Use wrap_in_articulation=False and add explicit "
                            "closure constraints if you need a ring/loop.",
                            UserWarning,
                            stacklevel=2,
                        )

                # Wrap the connected component into an articulation.
                if component_joints:
                    if key:
                        art_key = (
                            f"{key}_articulation_{component_index}" if component_index > 0 else f"{key}_articulation"
                        )
                    else:
                        art_key = None
                    self.add_articulation(component_joints, key=art_key)

                component_index += 1

            return all_joints

        if not wrap_in_articulation:
            all_joints = _build_joints_star()
        else:
            all_joints = _build_joints_forest()

        if junction_collision_filter:
            # Filter collisions among *non-jointed* sibling bodies incident to each junction node
            # (degree >= 3). Jointed parent/child pairs are already filtered by
            # add_joint_cable(collision_filter_parent=True).
            for inc in node_incidence:
                if len(inc) < 3:
                    continue
                bodies_set = {edge_bodies[e_idx] for e_idx in inc}
                if len(bodies_set) < 2:
                    continue
                bodies = sorted(bodies_set)

                for i in range(len(bodies)):
                    for j in range(i + 1, len(bodies)):
                        bi = bodies[i]
                        bj = bodies[j]
                        if (bi, bj) in jointed_body_pairs:
                            # Already filtered by add_joint_cable(collision_filter_parent=True).
                            continue
                        for si in self.body_shapes.get(bi, []):
                            for sj in self.body_shapes.get(bj, []):
                                self.add_shape_collision_filter_pair(int(si), int(sj))

        return edge_bodies, all_joints

    # endregion

    # particles
    def add_particle(
        self,
        pos: Vec3,
        vel: Vec3,
        mass: float,
        radius: float | None = None,
        flags: int = ParticleFlags.ACTIVE,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a single particle to the model.

        Args:
            pos: The initial position of the particle.
            vel: The initial velocity of the particle.
            mass: The mass of the particle.
            radius: The radius of the particle used in collision handling. If None, the radius is set to the default value (:attr:`default_particle_radius`).
            flags: The flags that control the dynamical behavior of the particle, see :class:`newton.ParticleFlags`.
            custom_attributes: Dictionary of custom attribute names to values.

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that is not subject to dynamics.

        Returns:
            The index of the particle in the system.
        """
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)
        if radius is None:
            radius = self.default_particle_radius
        self.particle_radius.append(radius)
        self.particle_flags.append(flags)
        self.particle_world.append(self.current_world)

        particle_id = self.particle_count - 1

        # Process custom attributes
        if custom_attributes:
            self._process_custom_attributes(
                entity_index=particle_id,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.PARTICLE,
            )

        return particle_id

    def add_particles(
        self,
        pos: list[Vec3],
        vel: list[Vec3],
        mass: list[float],
        radius: list[float] | None = None,
        flags: list[int] | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ):
        """Adds a group of particles to the model.

        Args:
            pos: The initial positions of the particles.
            vel: The initial velocities of the particles.
            mass: The mass of the particles.
            radius: The radius of the particles used in collision handling. If None, the radius is set to the default value (:attr:`default_particle_radius`).
            flags: The flags that control the dynamical behavior of the particles, see :class:`newton.ParticleFlags`.
            custom_attributes: Dictionary of custom attribute names to lists of values (one value for each particle).

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that is not subject to dynamics.
        """
        particle_start = self.particle_count
        particle_count = len(pos)

        self.particle_q.extend(pos)
        self.particle_qd.extend(vel)
        self.particle_mass.extend(mass)
        if radius is None:
            radius = [self.default_particle_radius] * particle_count
        if flags is None:
            flags = [ParticleFlags.ACTIVE] * particle_count
        self.particle_radius.extend(radius)
        self.particle_flags.extend(flags)
        # Maintain world assignment for bulk particle creation
        self.particle_world.extend([self.current_world] * particle_count)

        # Process custom attributes
        if custom_attributes and particle_count:
            particle_indices = list(range(particle_start, particle_start + particle_count))
            self._process_custom_attributes(
                entity_index=particle_indices,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.PARTICLE,
            )

    def add_spring(
        self,
        i: int,
        j: int,
        ke: float,
        kd: float,
        control: float,
        custom_attributes: dict[str, Any] | None = None,
    ):
        """Adds a spring between two particles in the system

        Args:
            i: The index of the first particle
            j: The index of the second particle
            ke: The elastic stiffness of the spring
            kd: The damping stiffness of the spring
            control: The actuation level of the spring
            custom_attributes: Dictionary of custom attribute names to values.

        Note:
            The spring is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        self.spring_indices.append(i)
        self.spring_indices.append(j)
        self.spring_stiffness.append(ke)
        self.spring_damping.append(kd)
        self.spring_control.append(control)

        # compute rest length
        p = self.particle_q[i]
        q = self.particle_q[j]

        delta = np.subtract(p, q)
        l = np.sqrt(np.dot(delta, delta))

        self.spring_rest_length.append(l)

        # Process custom attributes
        if custom_attributes:
            spring_index = len(self.spring_rest_length) - 1
            self._process_custom_attributes(
                entity_index=spring_index,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.SPRING,
            )

    def add_triangle(
        self,
        i: int,
        j: int,
        k: int,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> float:
        """Adds a triangular FEM element between three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        parameters specified on the model. See :attr:`~newton.Model.tri_ke`, :attr:`~newton.Model.tri_kd`.

        Args:
            i: The index of the first particle.
            j: The index of the second particle.
            k: The index of the third particle.
            tri_ke: The elastic stiffness of the triangle. If None, the default value (:attr:`default_tri_ke`) is used.
            tri_ka: The area stiffness of the triangle. If None, the default value (:attr:`default_tri_ka`) is used.
            tri_kd: The damping stiffness of the triangle. If None, the default value (:attr:`default_tri_kd`) is used.
            tri_drag: The drag coefficient of the triangle. If None, the default value (:attr:`default_tri_drag`) is used.
            tri_lift: The lift coefficient of the triangle. If None, the default value (:attr:`default_tri_lift`) is used.
            custom_attributes: Dictionary of custom attribute names to values.

        Return:
            The area of the triangle

        Note:
            The triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.
        """
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        # compute basis for 2D rest pose
        p = self.particle_q[i]
        q = self.particle_q[j]
        r = self.particle_q[k]

        qp = q - p
        rp = r - p

        # construct basis aligned with the triangle
        n = wp.normalize(wp.cross(qp, rp))
        e1 = wp.normalize(qp)
        e2 = wp.normalize(wp.cross(n, e1))

        R = np.array((e1, e2))
        M = np.array((qp, rp))

        D = R @ M.T

        area = np.linalg.det(D) / 2.0

        if area <= 0.0:
            print("inverted or degenerate triangle element")
            return 0.0
        else:
            inv_D = np.linalg.inv(D)

            self.tri_indices.append((i, j, k))
            self.tri_poses.append(inv_D.tolist())
            self.tri_activations.append(0.0)
            self.tri_materials.append((tri_ke, tri_ka, tri_kd, tri_drag, tri_lift))
            self.tri_areas.append(area)

            # Process custom attributes
            if custom_attributes:
                tri_index = len(self.tri_indices) - 1
                self._process_custom_attributes(
                    entity_index=tri_index,
                    custom_attrs=custom_attributes,
                    expected_frequency=Model.AttributeFrequency.TRIANGLE,
                )
            return area

    def add_triangles(
        self,
        i: list[int] | nparray,
        j: list[int] | nparray,
        k: list[int] | nparray,
        tri_ke: list[float] | None = None,
        tri_ka: list[float] | None = None,
        tri_kd: list[float] | None = None,
        tri_drag: list[float] | None = None,
        tri_lift: list[float] | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> list[float]:
        """Adds triangular FEM elements between groups of three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        Parameters specified on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The indices of the first particle
            j: The indices of the second particle
            k: The indices of the third particle
            tri_ke: The elastic stiffness of the triangles. If None, the default value (:attr:`default_tri_ke`) is used.
            tri_ka: The area stiffness of the triangles. If None, the default value (:attr:`default_tri_ka`) is used.
            tri_kd: The damping stiffness of the triangles. If None, the default value (:attr:`default_tri_kd`) is used.
            tri_drag: The drag coefficient of the triangles. If None, the default value (:attr:`default_tri_drag`) is used.
            tri_lift: The lift coefficient of the triangles. If None, the default value (:attr:`default_tri_lift`) is used.
            custom_attributes: Dictionary of custom attribute names to values.

        Return:
            The areas of the triangles

        Note:
            A triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        # compute basis for 2D rest pose
        q_ = np.asarray(self.particle_q)
        p = q_[i]
        q = q_[j]
        r = q_[k]

        qp = q - p
        rp = r - p

        def normalized(a):
            l = np.linalg.norm(a, axis=-1, keepdims=True)
            l[l == 0] = 1.0
            return a / l

        n = normalized(np.cross(qp, rp))
        e1 = normalized(qp)
        e2 = normalized(np.cross(n, e1))

        R = np.concatenate((e1[..., None], e2[..., None]), axis=-1)
        M = np.concatenate((qp[..., None], rp[..., None]), axis=-1)

        D = np.matmul(R.transpose(0, 2, 1), M)

        areas = np.linalg.det(D) / 2.0
        areas[areas < 0.0] = 0.0
        valid_inds = (areas > 0.0).nonzero()[0]
        if len(valid_inds) < len(areas):
            print("inverted or degenerate triangle elements")

        D[areas == 0.0] = np.eye(2)[None, ...]
        inv_D = np.linalg.inv(D)

        i_ = np.asarray(i)
        j_ = np.asarray(j)
        k_ = np.asarray(k)

        inds = np.concatenate((i_[valid_inds, None], j_[valid_inds, None], k_[valid_inds, None]), axis=-1)

        tri_start = len(self.tri_indices)
        self.tri_indices.extend(inds.tolist())
        self.tri_poses.extend(inv_D[valid_inds].tolist())
        self.tri_activations.extend([0.0] * len(valid_inds))

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(areas)
            return arr

        tri_ke = init_if_none(tri_ke, self.default_tri_ke)
        tri_ka = init_if_none(tri_ka, self.default_tri_ka)
        tri_kd = init_if_none(tri_kd, self.default_tri_kd)
        tri_drag = init_if_none(tri_drag, self.default_tri_drag)
        tri_lift = init_if_none(tri_lift, self.default_tri_lift)

        self.tri_materials.extend(
            zip(
                np.array(tri_ke)[valid_inds],
                np.array(tri_ka)[valid_inds],
                np.array(tri_kd)[valid_inds],
                np.array(tri_drag)[valid_inds],
                np.array(tri_lift)[valid_inds],
                strict=False,
            )
        )
        areas = areas.tolist()
        self.tri_areas.extend(areas)

        # Process custom attributes
        if custom_attributes and len(valid_inds) > 0:
            tri_indices = list(range(tri_start, tri_start + len(valid_inds)))
            self._process_custom_attributes(
                entity_index=tri_indices,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.TRIANGLE,
            )
        return areas

    def add_tetrahedron(
        self,
        i: int,
        j: int,
        k: int,
        l: int,
        k_mu: float = 1.0e3,
        k_lambda: float = 1.0e3,
        k_damp: float = 0.0,
        custom_attributes: dict[str, Any] | None = None,
    ) -> float:
        """Adds a tetrahedral FEM element between four particles in the system.

        Tetrahedra are modeled as viscoelastic elements with a NeoHookean energy
        density based on [Smith et al. 2018].

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle
            l: The index of the fourth particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The element's damping stiffness
            custom_attributes: Dictionary of custom attribute names to values.

        Return:
            The volume of the tetrahedron

        Note:
            The tetrahedron is created with a rest-pose based on the particle's initial configuration

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q[i])
        q = np.array(self.particle_q[j])
        r = np.array(self.particle_q[k])
        s = np.array(self.particle_q[l])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.array((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if volume <= 0.0:
            print("inverted tetrahedral element")
        else:
            inv_Dm = np.linalg.inv(Dm)

            self.tet_indices.append((i, j, k, l))
            self.tet_poses.append(inv_Dm.tolist())
            self.tet_activations.append(0.0)
            self.tet_materials.append((k_mu, k_lambda, k_damp))

            # Process custom attributes
            if custom_attributes:
                tet_index = len(self.tet_indices) - 1
                self._process_custom_attributes(
                    entity_index=tet_index,
                    custom_attrs=custom_attributes,
                    expected_frequency=Model.AttributeFrequency.TETRAHEDRON,
                )

        return volume

    def add_edge(
        self,
        i: int,
        j: int,
        k: int,
        l: int,
        rest: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """Adds a bending edge element between two adjacent triangles in the cloth mesh, defined by four vertices.

        The bending energy model follows the discrete shell formulation from [Grinspun et al. 2003].
        The bending stiffness is controlled by the `edge_ke` parameter, and the bending damping by the `edge_kd` parameter.

        Args:
            i: The index of the first particle, i.e., opposite vertex 0
            j: The index of the second particle, i.e., opposite vertex 1
            k: The index of the third particle, i.e., vertex 0
            l: The index of the fourth particle, i.e., vertex 1
            rest: The rest angle across the edge in radians, if not specified it will be computed
            edge_ke: The bending stiffness coefficient
            edge_kd: The bending damping coefficient
            custom_attributes: Dictionary of custom attribute names to values.

        Return:
            The index of the edge.

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counterclockwise
            winding: (i, k, l), (j, l, k).

        """
        edge_ke = edge_ke if edge_ke is not None else self.default_edge_ke
        edge_kd = edge_kd if edge_kd is not None else self.default_edge_kd

        # compute rest angle
        x3 = self.particle_q[k]
        x4 = self.particle_q[l]
        if rest is None:
            rest = 0.0
            if i != -1 and j != -1:
                x1 = self.particle_q[i]
                x2 = self.particle_q[j]

                n1 = wp.normalize(wp.cross(x3 - x1, x4 - x1))
                n2 = wp.normalize(wp.cross(x4 - x2, x3 - x2))
                e = wp.normalize(x4 - x3)

                cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
                sin_theta = np.dot(np.cross(n1, n2), e)
                rest = math.atan2(sin_theta, cos_theta)

        self.edge_indices.append((i, j, k, l))
        self.edge_rest_angle.append(rest)
        self.edge_rest_length.append(wp.length(x4 - x3))
        self.edge_bending_properties.append((edge_ke, edge_kd))
        edge_index = len(self.edge_indices) - 1

        # Process custom attributes
        if custom_attributes:
            self._process_custom_attributes(
                entity_index=edge_index,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.EDGE,
            )

        return edge_index

    def add_edges(
        self,
        i: list[int],
        j: list[int],
        k: list[int],
        l: list[int],
        rest: list[float] | None = None,
        edge_ke: list[float] | None = None,
        edge_kd: list[float] | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ) -> None:
        """Adds bending edge elements between two adjacent triangles in the cloth mesh, defined by four vertices.

        The bending energy model follows the discrete shell formulation from [Grinspun et al. 2003].
        The bending stiffness is controlled by the `edge_ke` parameter, and the bending damping by the `edge_kd` parameter.

        Args:
            i: The indices of the first particles, i.e., opposite vertex 0
            j: The indices of the second particles, i.e., opposite vertex 1
            k: The indices of the third particles, i.e., vertex 0
            l: The indices of the fourth particles, i.e., vertex 1
            rest: The rest angles across the edges in radians, if not specified they will be computed
            edge_ke: The bending stiffness coefficients
            edge_kd: The bending damping coefficients
            custom_attributes: Dictionary of custom attribute names to values.

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counterclockwise
            winding: (i, k, l), (j, l, k).

        """
        # Convert inputs to numpy arrays
        i_ = np.asarray(i)
        j_ = np.asarray(j)
        k_ = np.asarray(k)
        l_ = np.asarray(l)

        # Cache particle positions as numpy array
        particle_q_ = np.asarray(self.particle_q)
        x3 = particle_q_[k_]
        x4 = particle_q_[l_]
        x4_minus_x3 = x4 - x3

        if rest is None:
            rest = np.zeros_like(i_, dtype=float)
            valid_mask = (i_ != -1) & (j_ != -1)

            # compute rest angle
            x1_valid = particle_q_[i_[valid_mask]]
            x2_valid = particle_q_[j_[valid_mask]]
            x3_valid = particle_q_[k_[valid_mask]]
            x4_valid = particle_q_[l_[valid_mask]]

            def normalized(a):
                l = np.linalg.norm(a, axis=-1, keepdims=True)
                l[l == 0] = 1.0
                return a / l

            n1 = normalized(np.cross(x3_valid - x1_valid, x4_valid - x1_valid))
            n2 = normalized(np.cross(x4_valid - x2_valid, x3_valid - x2_valid))
            e = normalized(x4_valid - x3_valid)

            def dot(a, b):
                return (a * b).sum(axis=-1)

            cos_theta = np.clip(dot(n1, n2), -1.0, 1.0)
            sin_theta = dot(np.cross(n1, n2), e)
            rest[valid_mask] = np.arctan2(sin_theta, cos_theta)
            rest = rest.tolist()

        inds = np.concatenate((i_[:, None], j_[:, None], k_[:, None], l_[:, None]), axis=-1)

        edge_start = len(self.edge_indices)
        self.edge_indices.extend(inds.tolist())
        self.edge_rest_angle.extend(rest)
        self.edge_rest_length.extend(np.linalg.norm(x4_minus_x3, axis=1).tolist())

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(i)
            return arr

        edge_ke = init_if_none(edge_ke, self.default_edge_ke)
        edge_kd = init_if_none(edge_kd, self.default_edge_kd)

        self.edge_bending_properties.extend(zip(edge_ke, edge_kd, strict=False))

        # Process custom attributes
        if custom_attributes and len(i) > 0:
            edge_indices = list(range(edge_start, edge_start + len(i)))
            self._process_custom_attributes(
                entity_index=edge_indices,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.EDGE,
            )

    def add_cloth_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        cell_x: float,
        cell_y: float,
        mass: float,
        reverse_winding: bool = False,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
        add_springs: bool = False,
        spring_ke: float | None = None,
        spring_kd: float | None = None,
        particle_radius: float | None = None,
        custom_attributes_particles: dict[str, Any] | None = None,
        custom_attributes_edges: dict[str, Any] | None = None,
        custom_attributes_triangles: dict[str, Any] | None = None,
    ):
        """Helper to create a regular planar cloth grid

        Creates a rectangular grid of particles with FEM triangles and bending elements
        automatically.

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            mass: The mass of each particle
            reverse_winding: Flip the winding of the mesh
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
        """

        def grid_index(x, y, dim_x):
            return y * dim_x + x

        indices, vertices = [], []
        for y in range(0, dim_y + 1):
            for x in range(0, dim_x + 1):
                local_pos = wp.vec3(x * cell_x, y * cell_y, 0.0)
                vertices.append(local_pos)
                if x > 0 and y > 0:
                    v0 = grid_index(x - 1, y - 1, dim_x + 1)
                    v1 = grid_index(x, y - 1, dim_x + 1)
                    v2 = grid_index(x, y, dim_x + 1)
                    v3 = grid_index(x - 1, y, dim_x + 1)
                    if reverse_winding:
                        indices.extend([v0, v1, v2])
                        indices.extend([v0, v2, v3])
                    else:
                        indices.extend([v0, v1, v3])
                        indices.extend([v1, v2, v3])

        start_vertex = len(self.particle_q)

        total_mass = mass * (dim_x + 1) * (dim_x + 1)
        total_area = cell_x * cell_y * dim_x * dim_y
        density = total_mass / total_area

        self.add_cloth_mesh(
            pos=pos,
            rot=rot,
            scale=1.0,
            vel=vel,
            vertices=vertices,
            indices=indices,
            density=density,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            tri_drag=tri_drag,
            tri_lift=tri_lift,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            add_springs=add_springs,
            spring_ke=spring_ke,
            spring_kd=spring_kd,
            particle_radius=particle_radius,
            custom_attributes_particles=custom_attributes_particles,
            custom_attributes_triangles=custom_attributes_triangles,
            custom_attributes_edges=custom_attributes_edges,
        )

        vertex_id = 0
        for y in range(dim_y + 1):
            for x in range(dim_x + 1):
                particle_mass = mass
                particle_flag = ParticleFlags.ACTIVE

                if (
                    (x == 0 and fix_left)
                    or (x == dim_x and fix_right)
                    or (y == 0 and fix_bottom)
                    or (y == dim_y and fix_top)
                ):
                    particle_flag = particle_flag & ~ParticleFlags.ACTIVE
                    particle_mass = 0.0

                self.particle_flags[start_vertex + vertex_id] = particle_flag
                self.particle_mass[start_vertex + vertex_id] = particle_mass
                vertex_id = vertex_id + 1

    def add_cloth_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: list[Vec3],
        indices: list[int],
        density: float,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
        add_springs: bool = False,
        spring_ke: float | None = None,
        spring_kd: float | None = None,
        particle_radius: float | None = None,
        custom_attributes_particles: dict[str, Any] | None = None,
        custom_attributes_edges: dict[str, Any] | None = None,
        custom_attributes_triangles: dict[str, Any] | None = None,
        custom_attributes_springs: dict[str, Any] | None = None,
    ) -> None:
        """Helper to create a cloth model from a regular triangle mesh

        Creates one FEM triangle element and one bending element for every face
        and edge in the input triangle mesh

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            vertices: A list of vertex positions
            indices: A list of triangle indices, 3 entries per-face
            density: The density per-area of the mesh
            particle_radius: The particle_radius which controls particle based collisions.
            custom_attributes_particles: Dictionary of custom attribute names to values for the particles.
            custom_attributes_edges: Dictionary of custom attribute names to values for the edges.
            custom_attributes_triangles: Dictionary of custom attribute names to values for the triangles.
            custom_attributes_springs: Dictionary of custom attribute names to values for the springs.

        Note:
            The mesh should be two-manifold.
        """
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift
        edge_ke = edge_ke if edge_ke is not None else self.default_edge_ke
        edge_kd = edge_kd if edge_kd is not None else self.default_edge_kd
        spring_ke = spring_ke if spring_ke is not None else self.default_spring_ke
        spring_kd = spring_kd if spring_kd is not None else self.default_spring_kd
        particle_radius = particle_radius if particle_radius is not None else self.default_particle_radius

        num_verts = int(len(vertices))
        num_tris = int(len(indices) / 3)

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        # particles
        # for v in vertices:
        #     p = wp.quat_rotate(rot, v * scale) + pos
        #     self.add_particle(p, vel, 0.0, radius=particle_radius)
        vertices_np = np.array(vertices) * scale
        rot_mat_np = np.array(wp.quat_to_matrix(rot), dtype=np.float32).reshape(3, 3)
        verts_3d_np = np.dot(vertices_np, rot_mat_np.T) + pos
        self.add_particles(
            verts_3d_np.tolist(),
            [vel] * num_verts,
            mass=[0.0] * num_verts,
            radius=[particle_radius] * num_verts,
            custom_attributes=custom_attributes_particles,
        )

        # triangles
        inds = start_vertex + np.array(indices)
        inds = inds.reshape(-1, 3)
        areas = self.add_triangles(
            inds[:, 0],
            inds[:, 1],
            inds[:, 2],
            [tri_ke] * num_tris,
            [tri_ka] * num_tris,
            [tri_kd] * num_tris,
            [tri_drag] * num_tris,
            [tri_lift] * num_tris,
            custom_attributes=custom_attributes_triangles,
        )
        for t in range(num_tris):
            area = areas[t]

            self.particle_mass[inds[t, 0]] += density * area / 3.0
            self.particle_mass[inds[t, 1]] += density * area / 3.0
            self.particle_mass[inds[t, 2]] += density * area / 3.0

        end_tri = len(self.tri_indices)

        adj = MeshAdjacency(self.tri_indices[start_tri:end_tri])

        edge_indices = np.fromiter(
            (x for e in adj.edges.values() for x in (e.o0, e.o1, e.v0, e.v1)),
            int,
        ).reshape(-1, 4)
        self.add_edges(
            edge_indices[:, 0],
            edge_indices[:, 1],
            edge_indices[:, 2],
            edge_indices[:, 3],
            edge_ke=[edge_ke] * len(edge_indices),
            edge_kd=[edge_kd] * len(edge_indices),
            custom_attributes=custom_attributes_edges,
        )

        if add_springs:
            spring_indices = set()
            for i, j, k, l in edge_indices:
                spring_indices.add((min(k, l), max(k, l)))
                if i != -1:
                    spring_indices.add((min(i, k), max(i, k)))
                    spring_indices.add((min(i, l), max(i, l)))
                if j != -1:
                    spring_indices.add((min(j, k), max(j, k)))
                    spring_indices.add((min(j, l), max(j, l)))
                if i != -1 and j != -1:
                    spring_indices.add((min(i, j), max(i, j)))

            for i, j in spring_indices:
                self.add_spring(i, j, spring_ke, spring_kd, control=0.0, custom_attributes=custom_attributes_springs)

    def add_particle_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        mass: float,
        jitter: float,
        radius_mean: float | None = None,
        radius_std: float = 0.0,
        flags: list[int] | int | None = None,
        custom_attributes: dict[str, Any] | None = None,
    ):
        """
        Adds a regular 3D grid of particles to the model.

        This helper function creates a grid of particles arranged in a rectangular lattice,
        with optional random jitter and per-particle radius variation. The grid is defined
        by its dimensions along each axis and the spacing between particles.

        Args:
            pos: The world-space position of the grid origin.
            rot: The rotation to apply to the grid (as a quaternion).
            vel: The initial velocity to assign to each particle.
            dim_x: Number of particles along the X axis.
            dim_y: Number of particles along the Y axis.
            dim_z: Number of particles along the Z axis.
            cell_x: Spacing between particles along the X axis.
            cell_y: Spacing between particles along the Y axis.
            cell_z: Spacing between particles along the Z axis.
            mass: Mass to assign to each particle.
            jitter: Maximum random offset to apply to each particle position.
            radius_mean: Mean radius for particles. If None, uses the builder's default.
            radius_std: Standard deviation for particle radii. If > 0, radii are sampled from a normal distribution.
            flags: Flags to assign to each particle. If None, uses the builder's default.
            custom_attributes: Dictionary of custom attribute names to values for the particles.

        Returns:
            None
        """

        # local grid
        px = np.arange(dim_x) * cell_x
        py = np.arange(dim_y) * cell_y
        pz = np.arange(dim_z) * cell_z
        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        # apply transform to points
        rot_mat = wp.quat_to_matrix(rot)
        points = points @ np.array(rot_mat).reshape(3, 3).T + np.array(pos)
        velocity = np.broadcast_to(np.array(vel).reshape(1, 3), points.shape)

        # add jitter
        rng = np.random.default_rng(42 + len(self.particle_q))
        points += (rng.random(points.shape) - 0.5) * jitter

        if radius_mean is None:
            radius_mean = self.default_particle_radius

        radii = np.full(points.shape[0], fill_value=radius_mean)
        if radius_std > 0.0:
            radii += rng.standard_normal(radii.shape) * radius_std

        masses = [mass] * points.shape[0]
        if flags is not None:
            flags = [flags] * points.shape[0]

        # Broadcast scalar custom attribute values to all particles
        num_particles = points.shape[0]
        broadcast_custom_attrs = None
        if custom_attributes:
            broadcast_custom_attrs = {}
            for key, value in custom_attributes.items():
                # Check if value is a sequence (but not string/bytes) or numpy array
                is_array = isinstance(value, np.ndarray)
                is_sequence = isinstance(value, Sequence) and not isinstance(value, (str, bytes))

                if is_array or is_sequence:
                    # Value is already a sequence/array - validate length
                    if len(value) != num_particles:
                        raise ValueError(
                            f"Custom attribute '{key}' has {len(value)} values but {num_particles} particles in grid"
                        )
                    broadcast_custom_attrs[key] = list(value) if is_array else value
                else:
                    # Scalar value - broadcast to all particles
                    broadcast_custom_attrs[key] = [value] * num_particles

        self.add_particles(
            pos=points.tolist(),
            vel=velocity.tolist(),
            mass=masses,
            radius=radii.tolist(),
            flags=flags,
            custom_attributes=broadcast_custom_attrs,
        )

    def add_soft_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        tri_ke: float = 0.0,
        tri_ka: float = 0.0,
        tri_kd: float = 0.0,
        tri_drag: float = 0.0,
        tri_lift: float = 0.0,
        add_surface_mesh_edges: bool = True,
        edge_ke: float = 0.0,
        edge_kd: float = 0.0,
        particle_radius: float | None = None,
    ):
        """Helper to create a rectangular tetrahedral FEM grid

        Creates a regular grid of FEM tetrahedra and surface triangles. Useful for example
        to create beams and sheets. Each hexahedral cell is decomposed into 5
        tetrahedral elements.

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            dim_z: The number of rectangular cells along the z-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            cell_z: The width of each cell in the z-direction
            density: The density of each particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
            tri_ke: Stiffness for surface mesh triangles. Defaults to 0.0.
            tri_ka: Area stiffness for surface mesh triangles. Defaults to 0.0.
            tri_kd: Damping for surface mesh triangles. Defaults to 0.0.
            tri_drag: Drag coefficient for surface mesh triangles. Defaults to 0.0.
            tri_lift: Lift coefficient for surface mesh triangles. Defaults to 0.0.
            add_surface_mesh_edges: Whether to create zero-stiffness bending edges on the
                generated surface mesh. These edges improve collision robustness for VBD solver. Defaults to True.
            edge_ke: Bending edge stiffness used when ``add_surface_mesh_edges`` is True. Defaults to 0.0.
            edge_kd: Bending edge damping used when ``add_surface_mesh_edges`` is True. Defaults to 0.0.
            particle_radius: particle's contact radius (controls rigidbody-particle contact distance)

        Note:
            The generated surface triangles and optional edges are for collision purposes.
            Their stiffness and damping values default to zero so they do not introduce additional
            elastic forces. Set the triangle stiffness parameters above to non-zero values if you
            want the surface to behave like a thin skin.
        """
        start_vertex = len(self.particle_q)

        mass = cell_x * cell_y * cell_z * density

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    if fix_left and x == 0:
                        m = 0.0

                    if fix_right and x == dim_x:
                        m = 0.0

                    if fix_top and y == dim_y:
                        m = 0.0

                    if fix_bottom and y == 0:
                        m = 0.0

                    p = wp.quat_rotate(rot, v) + pos

                    self.add_particle(p, vel, m, particle_radius)

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = grid_index(x, y, z) + start_vertex
                    v1 = grid_index(x + 1, y, z) + start_vertex
                    v2 = grid_index(x + 1, y, z + 1) + start_vertex
                    v3 = grid_index(x, y, z + 1) + start_vertex
                    v4 = grid_index(x, y + 1, z) + start_vertex
                    v5 = grid_index(x + 1, y + 1, z) + start_vertex
                    v6 = grid_index(x + 1, y + 1, z + 1) + start_vertex
                    v7 = grid_index(x, y + 1, z + 1) + start_vertex

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:
                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)

        # add surface triangles
        start_tri = len(self.tri_indices)
        for _k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
        end_tri = len(self.tri_indices)

        if add_surface_mesh_edges:
            # add surface mesh edges (for collision)
            if end_tri > start_tri:
                adj = MeshAdjacency(self.tri_indices[start_tri:end_tri])
                edge_indices = np.fromiter(
                    (x for e in adj.edges.values() for x in (e.o0, e.o1, e.v0, e.v1)),
                    int,
                ).reshape(-1, 4)
                if len(edge_indices) > 0:
                    # Add edges with specified stiffness/damping (for collision)
                    for o1, o2, v1, v2 in edge_indices:
                        self.add_edge(o1, o2, v1, v2, None, edge_ke, edge_kd)

    def add_soft_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: list[Vec3],
        indices: list[int],
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        tri_ke: float = 0.0,
        tri_ka: float = 0.0,
        tri_kd: float = 0.0,
        tri_drag: float = 0.0,
        tri_lift: float = 0.0,
        add_surface_mesh_edges: bool = True,
        edge_ke: float = 0.0,
        edge_kd: float = 0.0,
        particle_radius: float | None = None,
    ) -> None:
        """Helper to create a tetrahedral model from an input tetrahedral mesh

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            vertices: A list of vertex positions, array of 3D points
            indices: A list of tetrahedron indices, 4 entries per-element, flattened array
            density: The density per-area of the mesh
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
            tri_ke: Stiffness for surface mesh triangles. Defaults to 0.0.
            tri_ka: Area stiffness for surface mesh triangles. Defaults to 0.0.
            tri_kd: Damping for surface mesh triangles. Defaults to 0.0.
            tri_drag: Drag coefficient for surface mesh triangles. Defaults to 0.0.
            tri_lift: Lift coefficient for surface mesh triangles. Defaults to 0.0.
            add_surface_mesh_edges: Whether to create zero-stiffness bending edges on the
                generated surface mesh. These edges improve collision robustness for VBD solver. Defaults to True.
            edge_ke: Bending edge stiffness used when ``add_surface_mesh_edges`` is True. Defaults to 0.0.
            edge_kd: Bending edge damping used when ``add_surface_mesh_edges`` is True. Defaults to 0.0.
            particle_radius: particle's contact radius (controls rigidbody-particle contact distance)

        Note:
            The generated surface triangles and optional edges are for collision purposes.
            Their stiffness and damping values default to zero so they do not introduce additional
            elastic forces. Set the stiffness parameters above to non-zero values if you
            want the surface to behave like a thin skin.
        """
        num_tets = int(len(indices) / 4)

        start_vertex = len(self.particle_q)

        # dict of open faces
        faces = {}

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        pos = wp.vec3(pos[0], pos[1], pos[2])
        # add particles
        for v in vertices:
            p = wp.quat_rotate(rot, wp.vec3(v[0], v[1], v[2]) * scale) + pos

            self.add_particle(p, vel, 0.0, particle_radius)

        # add tetrahedra
        for t in range(num_tets):
            v0 = start_vertex + indices[t * 4 + 0]
            v1 = start_vertex + indices[t * 4 + 1]
            v2 = start_vertex + indices[t * 4 + 2]
            v3 = start_vertex + indices[t * 4 + 3]

            volume = self.add_tetrahedron(v0, v1, v2, v3, k_mu, k_lambda, k_damp)

            # distribute volume fraction to particles
            if volume > 0.0:
                self.particle_mass[v0] += density * volume / 4.0
                self.particle_mass[v1] += density * volume / 4.0
                self.particle_mass[v2] += density * volume / 4.0
                self.particle_mass[v3] += density * volume / 4.0

                # build open faces
                add_face(v0, v2, v1)
                add_face(v1, v2, v3)
                add_face(v0, v1, v3)
                add_face(v0, v3, v2)

        # add surface triangles
        start_tri = len(self.tri_indices)
        for _k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
        end_tri = len(self.tri_indices)

        if add_surface_mesh_edges:
            # add surface mesh edges (for collision)
            if end_tri > start_tri:
                adj = MeshAdjacency(self.tri_indices[start_tri:end_tri])
                edge_indices = np.fromiter(
                    (x for e in adj.edges.values() for x in (e.o0, e.o1, e.v0, e.v1)),
                    int,
                ).reshape(-1, 4)
                if len(edge_indices) > 0:
                    # Add edges with specified stiffness/damping (for collision)
                    for o1, o2, v1, v2 in edge_indices:
                        self.add_edge(o1, o2, v1, v2, None, edge_ke, edge_kd)

    # incrementally updates rigid body mass with additional mass and inertia expressed at a local to the body
    def _update_body_mass(self, i, m, I, p, q):
        if i == -1:
            return

        # find new COM
        new_mass = self.body_mass[i] + m

        if new_mass == 0.0:  # no mass
            return

        new_com = (self.body_com[i] * self.body_mass[i] + p * m) / new_mass

        # shift inertia to new COM
        com_offset = new_com - self.body_com[i]
        shape_offset = new_com - p

        new_inertia = transform_inertia(
            self.body_mass[i], self.body_inertia[i], com_offset, wp.quat_identity()
        ) + transform_inertia(m, I, shape_offset, q)

        self.body_mass[i] = new_mass
        self.body_inertia[i] = new_inertia
        self.body_com[i] = new_com

        if new_mass > 0.0:
            self.body_inv_mass[i] = 1.0 / new_mass
        else:
            self.body_inv_mass[i] = 0.0

        if any(x for x in new_inertia):
            self.body_inv_inertia[i] = wp.inverse(new_inertia)
        else:
            self.body_inv_inertia[i] = new_inertia

    def _validate_parent_body(self, parent_body: int, child: int) -> None:
        """
        Validate that parent_body is a valid body index.

        Args:
            parent_body: The parent body index to validate (-1 for world is OK).
            child: The child body index (to check for self-reference).

        Raises:
            ValueError: If validation fails.
        """
        if parent_body == -1:
            return  # -1 is valid (world reference)

        # Check bounds
        if parent_body < -1:
            raise ValueError(f"Invalid parent_body index: {parent_body}. Must be >= -1 (use -1 for world).")

        if parent_body >= len(self.body_mass):
            raise ValueError(
                f"Invalid parent_body index: {parent_body}. "
                f"Body index out of bounds (model has {len(self.body_mass)} bodies)."
            )

        # Check self-reference
        if parent_body == child:
            raise ValueError(f"Cannot attach body {child} to itself (parent_body == child).")

        # Validate body has positive mass (optional warning)
        if self.body_mass[parent_body] <= 0.0:
            warnings.warn(
                f"parent_body {parent_body} has zero or negative mass ({self.body_mass[parent_body]}). "
                f"This may cause unexpected behavior.",
                UserWarning,
                stacklevel=3,
            )

    def _find_articulation_for_body(self, body_id: int) -> int | None:
        """
        Find which articulation (if any) contains the given body.

        A body "belongs to" the articulation where it appears as a child in a joint.
        If a body is only a parent (e.g., root body of an articulation), it belongs
        to the articulation of its child joints.

        Args:
            body_id: The body index to search for.

        Returns:
            int: Articulation index if found, or None if body is not in any articulation.

        Algorithm:
            1. Priority 1: Find articulation where body is a child (most common case)
               - A body can only be a child in ONE joint (tree structure)
               - This uniquely identifies the body's home articulation
            2. Priority 2: Find articulation where body is a parent (for root bodies)
               - Root bodies are parents but not children
               - If parent in multiple articulations, returns the first found
               - This should be rare; most bodies are children in exactly one articulation

        Note:
            In valid tree structures, a body should be a child in at most one joint,
            making this lookup deterministic. Bodies that are only parents (root bodies)
            may appear in multiple articulations; in such cases, the first articulation
            found is returned.
        """
        # Priority 1: Check if body is a child in any joint
        # A body should be a child in at most ONE joint (tree structure)
        for joint_idx in range(len(self.joint_child)):
            if self.joint_child[joint_idx] == body_id:
                art_id = self.joint_articulation[joint_idx]
                if art_id >= 0:  # -1 means no articulation
                    return art_id  # Body found as child - this is its home articulation

        # Priority 2: If not found as child, check if body is a parent in any joint
        # This handles root bodies that are parents but not children
        parent_articulations = []
        for joint_idx in range(len(self.joint_parent)):
            if self.joint_parent[joint_idx] == body_id:
                art_id = self.joint_articulation[joint_idx]
                if art_id >= 0 and art_id not in parent_articulations:
                    parent_articulations.append(art_id)

        # Use first articulation found, but warn if multiple (shouldn't happen in valid trees)
        if parent_articulations:
            result = parent_articulations[0]
            if len(parent_articulations) > 1:
                warnings.warn(
                    f"Body {body_id} is a parent in multiple articulations {parent_articulations}. "
                    f"Using articulation {result}. This may indicate an unusual model structure.",
                    UserWarning,
                    stacklevel=3,
                )
            return result

        return None

    @staticmethod
    def _validate_base_joint_params(floating: bool | None, base_joint: dict | None, parent: int) -> None:
        """
        Validate floating and base_joint parameter combinations.

        This is a shared validation function used by all importers (MJCF, URDF, USD)
        to ensure consistent parameter validation.

        Args:
            floating: The floating parameter value (True, False, or None).
            base_joint: Dict with joint parameters (or None).
            parent: The parent body index (-1 for world, >= 0 for a body).

        Raises:
            ValueError: If parameter combinations are invalid:
                - Both floating and base_joint are specified (mutually exclusive)
                - floating=True with parent != -1 (FREE joints require world frame)
                - base_joint dict contains conflicting keys like 'parent', 'child', etc.
        """
        if floating is not None and base_joint is not None:
            raise ValueError(
                f"Cannot specify both 'floating' and 'base_joint' parameters. "
                f"These are mutually exclusive ways to control root attachment:\n"
                f"  - Use 'floating' for simple FREE/FIXED joints\n"
                f"  - Use 'base_joint' dict for custom joint parameters\n"
                f"Current values: floating={floating}, base_joint={{dict}}"
            )

        if floating is True and parent != -1:
            raise ValueError(
                f"Cannot create FREE joint when parent_body={parent} (not world). "
                f"FREE joints must connect to world frame (parent_body=-1).\n"
                f"Did you mean:\n"
                f"  - Use floating=False to create FIXED joint to parent\n"
                f"  - Use base_joint dict with D6 joint parameters for 6-DOF mobility attached to parent"
            )

        # Validate base_joint dict doesn't contain conflicting keys
        if base_joint is not None:
            conflicting_keys = set(base_joint.keys()) & {"parent", "child", "parent_xform", "child_xform"}
            if conflicting_keys:
                raise ValueError(
                    f"base_joint dict cannot specify {conflicting_keys}. "
                    f"These parameters are automatically set based on parent_body and attachment:\n"
                    f"  - 'parent' is set from parent_body parameter (currently {parent})\n"
                    f"  - 'child' is set to the imported root body\n"
                    f"  - 'parent_xform' and 'child_xform' are set from xform parameter\n"
                    f"Please remove {conflicting_keys} from the base_joint dict and use the "
                    f"parent_body argument instead."
                )

    def _check_sequential_composition(self, parent_body: int) -> int | None:
        """
        Check if attaching to parent_body is sequential (most recent articulation).

        Args:
            parent_body: The parent body index to check.

        Returns:
            The parent articulation ID, or None if parent_body is world (-1) or not in an articulation.

        Raises:
            ValueError: If attempting to attach to a non-sequential articulation.
        """
        if parent_body == -1:
            return None

        parent_articulation = self._find_articulation_for_body(parent_body)
        if parent_articulation is None:
            return None

        num_articulations = len(self.articulation_start)
        is_sequential = parent_articulation == num_articulations - 1

        if is_sequential:
            return parent_articulation
        else:
            body_name = self.body_key[parent_body] if parent_body < len(self.body_key) else f"#{parent_body}"
            raise ValueError(
                f"Cannot attach to parent_body {body_name} in articulation #{parent_articulation} "
                f"(most recent is #{num_articulations - 1}). "
                f"Attach to the most recently added articulation or build in order."
            )

    def _finalize_imported_articulation(
        self,
        joint_indices: list[int],
        parent_body: int,
        articulation_key: str | None = None,
        custom_attributes: dict | None = None,
    ) -> None:
        """
        Attach imported joints to parent articulation or create a new articulation.

        This helper method encapsulates the common logic used by all importers (MJCF, URDF, USD)
        for handling articulation creation after importing a model.

        Args:
            joint_indices: List of joint indices from the imported model.
            parent_body: The parent body index (-1 for world, or a body index for hierarchical composition).
            articulation_key: Optional key for the articulation (e.g., model name).
            custom_attributes: Optional custom attributes for the articulation.

        Note:
            - If parent_body != -1 and it belongs to an articulation, the imported joints are added
              to the parent's articulation (only works for sequential composition).
            - If parent_body != -1 but is not in any articulation, raises ValueError.
            - If parent_body == -1, a new articulation is created.
            - If joint_indices is empty, does nothing.

        Raises:
            ValueError: If parent_body is specified but not part of any articulation.
        """
        if not joint_indices:
            return

        if parent_body != -1:
            # Check if attachment is sequential
            parent_articulation = self._check_sequential_composition(parent_body=parent_body)

            if parent_articulation is not None:
                # Mark all new joints as belonging to the parent's articulation
                for joint_idx in joint_indices:
                    self.joint_articulation[joint_idx] = parent_articulation
            else:
                # Parent body exists but is not in any articulation - this is an error
                # because user explicitly specified parent_body but it can't be used
                body_name = self.body_key[parent_body] if parent_body < len(self.body_key) else f"#{parent_body}"
                raise ValueError(
                    f"Cannot attach to parent_body '{body_name}': body is not part of any articulation. "
                    f"Only bodies within articulations can be used as parent_body. "
                    f"To create an independent articulation, use parent_body=-1 (default)."
                )
        else:
            # No parent_body specified, create a new articulation
            self.add_articulation(
                joints=joint_indices,
                key=articulation_key,
                custom_attributes=custom_attributes,
            )

    def _add_base_joint(
        self,
        child: int,
        floating: bool | None = None,
        base_joint: dict | None = None,
        key: str | None = None,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        parent: int = -1,
    ) -> int:
        """
        Internal helper for importers to create base joints.

        This method is used by importers (URDF, MJCF, USD) to attach imported bodies
        to the world or to a parent body with the appropriate joint type.

        Args:
            child: The body index to connect.
            floating: If None (default), behavior depends on format-specific defaults.
                If True, creates a FREE joint (only valid when ``parent == -1``).
                If False, always creates a fixed joint.
            base_joint: Dict with joint parameters passed to :meth:`add_joint`.
                Cannot be specified together with ``floating``.
            key: A unique key for the joint.
            parent_xform: The transform of the joint in the parent frame.
                If None, defaults to ``body_q[child]`` when parent is world (-1),
                or identity when parent is another body.
            child_xform: The transform of the joint in the child frame.
                If None, defaults to identity transform.
            parent: The index of the parent body. Use -1 (default) to connect to world.

        Returns:
            The index of the created joint.

        Raises:
            ValueError: If both ``floating`` and ``base_joint`` are specified,
                or if ``floating=True`` with ``parent != -1``, or if parent body
                is not part of any articulation.
        """
        # Validate parameter combinations
        self._validate_base_joint_params(floating, base_joint, parent)
        self._validate_parent_body(parent, child)

        # Validate that parent body is in an articulation (if not world)
        if parent != -1:
            parent_articulation = self._find_articulation_for_body(parent)
            if parent_articulation is None:
                body_name = self.body_key[parent] if parent < len(self.body_key) else f"#{parent}"
                raise ValueError(
                    f"Cannot attach to parent_body '{body_name}': body is not part of any articulation. "
                    f"Only bodies within articulations can be used as parent_body. "
                    f"To create an independent articulation, use parent_body=-1 (default)."
                )

        # Determine transforms
        if parent_xform is None:
            parent_xform = self.body_q[child] if parent == -1 else wp.transform_identity()
        if child_xform is None:
            child_xform = wp.transform_identity()

        # Create joint based on parameters
        if base_joint is not None:
            # Use custom joint parameters from dict
            joint_params = base_joint.copy()
            joint_params["parent"] = parent
            joint_params["child"] = child
            joint_params["parent_xform"] = parent_xform
            joint_params["child_xform"] = child_xform
            if "key" not in joint_params and key is not None:
                joint_params["key"] = key
            return self.add_joint(**joint_params)
        elif floating is True or (floating is None and parent == -1):
            # FREE joint (floating=True always requires parent==-1, validated above)
            # Note: We don't pass parent_xform here because add_joint_free initializes joint_q from body_q[child]
            # and the caller (e.g., URDF importer) will set the correct joint_q values afterward
            return self.add_joint_free(child, key=key)
        else:
            # FIXED joint (floating=False or floating=None with parent body)
            return self.add_joint_fixed(parent, child, parent_xform, child_xform, key=key)

    def _add_base_joints_to_floating_bodies(
        self,
        new_bodies: Iterable[int] | None = None,
        floating: bool | None = None,
        base_joint: dict | None = None,
    ):
        """
        Adds joints and single-joint articulations to every rigid body that is not a child in any joint
        and has positive mass.

        Args:
            new_bodies: The set of body indices to consider for adding joints.
            floating: If True or None (default), floating bodies receive a free joint.
                If False, floating bodies receive a fixed joint.
            base_joint: Dict with joint parameters passed to :meth:`add_joint`.
                When specified, this takes precedence over the ``floating`` parameter.

        Note:
            - Bodies that are already a child in any joint will be skipped.
            - Only bodies with strictly positive mass will receive a joint.
            - Each joint is added to its own single-joint articulation.
            - This is useful for ensuring that all floating (unconnected) bodies are properly articulated.
        """
        if new_bodies is None:
            return

        # set(self.joint_child) is connected_bodies
        floating_bodies = set(new_bodies) - set(self.joint_child)
        for body_id in floating_bodies:
            if self.body_mass[body_id] <= 0:
                continue

            joint = self._add_base_joint(body_id, floating=floating, base_joint=base_joint)
            # Use body key as articulation key for single-body articulations
            self.add_articulation([joint], key=self.body_key[body_id])

    def request_contact_attributes(self, *attributes: str) -> None:
        """
        Request that specific contact attributes be allocated when creating a Contacts object from the finalized Model.

        Args:
            *attributes: Variable number of attribute names (strings).
        """
        # Local import to avoid adding more module-level dependencies in this large file.
        from .contacts import Contacts  # noqa: PLC0415

        Contacts.validate_extended_attributes(attributes)
        self._requested_contact_attributes.update(attributes)

    def request_state_attributes(self, *attributes: str) -> None:
        """
        Request that specific state attributes be allocated when creating a State object from the finalized Model.

        See :ref:`extended_state_attributes` for details and usage.

        Args:
            *attributes: Variable number of attribute names (strings).
        """
        # Local import to avoid adding more module-level dependencies in this large file.
        from .state import State  # noqa: PLC0415

        State.validate_extended_attributes(attributes)
        self._requested_state_attributes.update(attributes)

    def set_coloring(self, particle_color_groups):
        """
        Sets coloring information with user-provided coloring.

        Args:
            particle_color_groups: A list of list or `np.array` with `dtype`=`int`. The length of the list is the number of colors
                and each list or `np.array` contains the indices of vertices with this color.
        """
        particle_color_groups = [
            color_group if isinstance(color_group, np.ndarray) else np.array(color_group)
            for color_group in particle_color_groups
        ]
        self.particle_color_groups = particle_color_groups

    def color(
        self,
        include_bending=False,
        balance_colors=True,
        target_max_min_color_ratio=1.1,
        coloring_algorithm=ColoringAlgorithm.MCS,
    ):
        """
        Runs coloring algorithm to generate coloring information.

        This populates both :attr:`particle_color_groups` (for particles) and
        :attr:`body_color_groups` (for rigid bodies) on the builder, which are
        consumed by :class:`newton.solvers.SolverVBD`.

        Call :meth:`color` (or :meth:`set_coloring`) before :meth:`finalize` when using
        :class:`newton.solvers.SolverVBD`; :meth:`finalize` does not implicitly color the model.

        Args:
            include_bending: Whether to include bending edges in the coloring graph. Set to `True` if your
                model contains bending edges (added via :meth:`add_edge`) that participate in bending constraints.
                When enabled, the coloring graph includes connections between opposite vertices of each edge (o1-o2),
                ensuring proper dependency handling for parallel bending computations. Leave as `False` if your model
                has no bending edges or if bending edges should not affect the coloring.
            balance_colors: Whether to apply the color balancing algorithm to balance the size of each color
            target_max_min_color_ratio: the color balancing algorithm will stop when the ratio between the largest color and
                the smallest color reaches this value
            algorithm: Value should be an enum type of ColoringAlgorithm, otherwise it will raise an error. ColoringAlgorithm.mcs means using the MCS coloring algorithm,
                while ColoringAlgorithm.ordered_greedy means using the degree-ordered greedy algorithm. The MCS algorithm typically generates 30% to 50% fewer colors
                compared to the ordered greedy algorithm, while maintaining the same linear complexity. Although MCS has a constant overhead that makes it about twice
                as slow as the greedy algorithm, it produces significantly better coloring results. We recommend using MCS, especially if coloring is only part of the
                preprocessing.

        Note:

            References to the coloring algorithm:

            MCS: Pereira, F. M. Q., & Palsberg, J. (2005, November). Register allocation via coloring of chordal graphs. In Asian Symposium on Programming Languages and Systems (pp. 315-329). Berlin, Heidelberg: Springer Berlin Heidelberg.

            Ordered Greedy: Ton-That, Q. M., Kry, P. G., & Andrews, S. (2023). Parallel block Neo-Hookean XPBD using graph clustering. Computers & Graphics, 110, 1-10.

        """
        if self.particle_count != 0:
            tri_indices = np.array(self.tri_indices, dtype=np.int32) if self.tri_indices else None
            tri_materials = np.array(self.tri_materials)
            tet_indices = np.array(self.tet_indices, dtype=np.int32) if self.tet_indices else None
            tet_materials = np.array(self.tet_materials)

            bending_edge_indices = None
            bending_edge_active_mask = None
            if include_bending and self.edge_indices:
                bending_edge_indices = np.array(self.edge_indices, dtype=np.int32)
                bending_edge_props = np.array(self.edge_bending_properties)
                # Active if either stiffness or damping is non-zero
                bending_edge_active_mask = (bending_edge_props[:, 0] != 0.0) | (bending_edge_props[:, 1] != 0.0)

            graph_edge_indices = construct_particle_graph(
                tri_indices,
                tri_materials[:, 0] * tri_materials[:, 1] if len(tri_materials) else None,
                bending_edge_indices,
                bending_edge_active_mask,
                tet_indices,
                tet_materials[:, 0] * tet_materials[:, 1] if len(tet_materials) else None,
            )

            if len(graph_edge_indices) > 0:
                self.particle_color_groups = color_graph(
                    self.particle_count,
                    graph_edge_indices,
                    balance_colors,
                    target_max_min_color_ratio,
                    coloring_algorithm,
                )
            else:
                # No edges to color - assign all particles to single color group
                if len(self.particle_q) > 0:
                    self.particle_color_groups = [np.arange(len(self.particle_q), dtype=int)]
                else:
                    self.particle_color_groups = []

        # Also color rigid bodies based on joint connectivity
        self.body_color_groups = color_rigid_bodies(
            self.body_count,
            self.joint_parent,
            self.joint_child,
            algorithm=coloring_algorithm,
            balance_colors=balance_colors,
            target_max_min_color_ratio=target_max_min_color_ratio,
        )

    def _validate_world_ordering(self):
        """Validate that world indices are monotonic, contiguous, and properly ordered.

        This method checks:
        1. World indices are monotonic (non-decreasing after first non-negative)
        2. World indices are contiguous (no gaps in sequence)
        3. Global entities (world -1) only appear at beginning or end of arrays
        4. All world indices are in valid range [-1, num_worlds-1]

        Raises:
            ValueError: If any validation check fails.
        """
        # List of all world arrays to validate
        world_arrays = [
            ("particle_world", self.particle_world),
            ("body_world", self.body_world),
            ("shape_world", self.shape_world),
            ("joint_world", self.joint_world),
            ("articulation_world", self.articulation_world),
            ("equality_constraint_world", self.equality_constraint_world),
            ("constraint_mimic_world", self.constraint_mimic_world),
        ]

        all_world_indices = set()

        for array_name, world_array in world_arrays:
            if not world_array:
                continue

            arr = np.array(world_array, dtype=np.int32)

            # Check for invalid world indices (must be in range [-1, num_worlds-1])
            max_valid = self.num_worlds - 1
            invalid_indices = np.where((arr < -1) | (arr > max_valid))[0]
            if len(invalid_indices) > 0:
                invalid_values = arr[invalid_indices]
                raise ValueError(
                    f"Invalid world index in {array_name}: found value(s) {invalid_values.tolist()} "
                    f"at indices {invalid_indices.tolist()}. Valid range is -1 to {max_valid} (num_worlds={self.num_worlds})."
                )

            # Check for global entity positioning (world -1)
            # Find first and last occurrence of -1
            negative_indices = np.where(arr == -1)[0]
            if len(negative_indices) > 0:
                # Check that all -1s form contiguous blocks at start and/or end
                # Count -1s at the start
                start_neg_count = 0
                for i in range(len(arr)):
                    if arr[i] == -1:
                        start_neg_count += 1
                    else:
                        break

                # Count -1s at the end (but only if they don't overlap with start)
                end_neg_count = 0
                if start_neg_count < len(arr):  # There are non-negative values after the start block
                    for i in range(len(arr) - 1, -1, -1):
                        if arr[i] == -1:
                            end_neg_count += 1
                        else:
                            break

                expected_neg_count = start_neg_count + end_neg_count
                actual_neg_count = len(negative_indices)

                if expected_neg_count != actual_neg_count:
                    # There are -1s in the middle
                    raise ValueError(
                        f"Invalid world ordering in {array_name}: global entities (world -1) "
                        f"must only appear at the beginning or end of the array, not in the middle. "
                        f"Found -1 values at indices: {negative_indices.tolist()}"
                    )

            # Check monotonic ordering for non-negative values
            non_neg_mask = arr >= 0
            if np.any(non_neg_mask):
                non_neg_values = arr[non_neg_mask]

                # Check that non-negative values are monotonic (non-decreasing)
                if not np.all(non_neg_values[1:] >= non_neg_values[:-1]):
                    # Find where the order breaks
                    for i in range(1, len(non_neg_values)):
                        if non_neg_values[i] < non_neg_values[i - 1]:
                            raise ValueError(
                                f"Invalid world ordering in {array_name}: world indices must be monotonic "
                                f"(non-decreasing). Found world {non_neg_values[i]} after world {non_neg_values[i - 1]}."
                            )

                # Collect all non-negative world indices for contiguity check
                all_world_indices.update(non_neg_values)

        # Check contiguity: all world indices should form a sequence 0, 1, 2, ..., n-1
        if all_world_indices:
            world_list = sorted(all_world_indices)
            expected = list(range(world_list[-1] + 1))

            if world_list != expected:
                missing = set(expected) - set(world_list)
                raise ValueError(
                    f"World indices are not contiguous. Missing world(s): {sorted(missing)}. "
                    f"Found worlds: {world_list}. Worlds must form a continuous sequence starting from 0."
                )

    def _validate_joints(self):
        """Validate that all joints belong to an articulation, except for "loop joints".

        Loop joints connect two bodies that are already reachable via articulated joints
        (used to create kinematic loops, converted to equality constraints by MuJoCo solver).

        Raises:
            ValueError: If any validation check fails.
        """
        if self.joint_count > 0:
            # First, find all bodies reachable via articulated joints
            articulated_bodies = set()
            articulated_bodies.add(-1)  # World is always reachable
            for i, art in enumerate(self.joint_articulation):
                if art >= 0:  # Joint is in an articulation
                    parent = self.joint_parent[i]
                    child = self.joint_child[i]
                    articulated_bodies.add(parent)
                    articulated_bodies.add(child)

            # Now check for true orphan joints: non-articulated joints whose child
            # is NOT reachable via other articulated joints
            orphan_joints = []
            for i, art in enumerate(self.joint_articulation):
                if art < 0:  # Joint is not in an articulation
                    child = self.joint_child[i]
                    if child not in articulated_bodies:
                        # This is a true orphan - the child body has no articulated path
                        orphan_joints.append(i)
                    # else: this is a loop joint - child is already reachable, so it's allowed

            if orphan_joints:
                joint_keys = [self.joint_key[i] for i in orphan_joints[:5]]  # Show first 5
                raise ValueError(
                    f"Found {len(orphan_joints)} joint(s) not belonging to any articulation. "
                    f"Call add_articulation() for all joints. Orphan joints: {joint_keys}"
                    + ("..." if len(orphan_joints) > 5 else "")
                )

    def _validate_shapes(self) -> bool:
        """Validate shape contact margins against thickness for stable broad phase detection.

        Thickness is an outward offset from a shape's surface, while AABBs are expanded
        by `contact_margin`. For reliable broad phase detection, each colliding shape
        should satisfy `contact_margin >= thickness` so that thickened surfaces produce
        overlapping AABBs.

        This check only considers shapes that participate in collisions (with the
        `COLLIDE_SHAPES` or `COLLIDE_PARTICLES` flag).

        Warns:
            UserWarning: If any colliding shape has `thickness > contact_margin`.

        Returns:
            bool: True if no shapes with thickness > contact_margin, False otherwise.
        """
        collision_flags_mask = ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES
        shapes_with_bad_margin = []
        for i in range(self.shape_count):
            # Skip shapes that don't participate in any collisions (e.g., sites, visual-only)
            if not (self.shape_flags[i] & collision_flags_mask):
                continue
            thickness = self.shape_thickness[i]
            margin = self.shape_contact_margin[i]
            if thickness > margin:
                shapes_with_bad_margin.append(
                    f"{self.shape_key[i] or f'shape_{i}'} (thickness={thickness:.6g}, margin={margin:.6g})"
                )
        if shapes_with_bad_margin:
            example_shapes = shapes_with_bad_margin[:5]
            warnings.warn(
                f"Found {len(shapes_with_bad_margin)} shape(s) with thickness > contact_margin. "
                f"This can cause missed collisions in broad phase since AABBs are only expanded by contact_margin. "
                f"Set contact_margin >= thickness for each shape. "
                f"Affected shapes: {example_shapes}" + ("..." if len(shapes_with_bad_margin) > 5 else ""),
                stacklevel=2,
            )
        return len(shapes_with_bad_margin) == 0

    def _validate_structure(self) -> None:
        """Validate structural invariants of the model.

        This method performs consolidated validation of all structural constraints,
        using vectorized numpy operations for efficiency:

        - Body references: shape_body, joint_parent, joint_child, equality_constraint_body1/2
        - Joint references: equality_constraint_joint1/2
        - Self-referential joints: joint_parent[i] != joint_child[i]
        - Start array monotonicity: joint_q_start, joint_qd_start, articulation_start
        - Array length consistency: per-DOF and per-coord arrays

        Raises:
            ValueError: If any structural validation check fails.
        """
        body_count = self.body_count
        joint_count = self.joint_count

        # Validate shape_body references: must be in [-1, body_count-1]
        if self.shape_count > 0:
            shape_body = np.array(self.shape_body, dtype=np.int32)
            invalid_mask = (shape_body < -1) | (shape_body >= body_count)
            if np.any(invalid_mask):
                invalid_indices = np.where(invalid_mask)[0]
                idx = invalid_indices[0]
                shape_key = self.shape_key[idx] or f"shape_{idx}"
                raise ValueError(
                    f"Invalid body reference in shape_body: shape {idx} ('{shape_key}') references body {shape_body[idx]}, "
                    f"but valid range is [-1, {body_count - 1}] (body_count={body_count})."
                )

        # Validate joint_parent references: must be in [-1, body_count-1]
        if joint_count > 0:
            joint_parent = np.array(self.joint_parent, dtype=np.int32)
            invalid_mask = (joint_parent < -1) | (joint_parent >= body_count)
            if np.any(invalid_mask):
                invalid_indices = np.where(invalid_mask)[0]
                idx = invalid_indices[0]
                joint_key = self.joint_key[idx] or f"joint_{idx}"
                raise ValueError(
                    f"Invalid body reference in joint_parent: joint {idx} ('{joint_key}') references parent body {joint_parent[idx]}, "
                    f"but valid range is [-1, {body_count - 1}] (body_count={body_count})."
                )

            # Validate joint_child references: must be in [0, body_count-1] (child cannot be world)
            joint_child = np.array(self.joint_child, dtype=np.int32)
            invalid_mask = (joint_child < 0) | (joint_child >= body_count)
            if np.any(invalid_mask):
                invalid_indices = np.where(invalid_mask)[0]
                idx = invalid_indices[0]
                joint_key = self.joint_key[idx] or f"joint_{idx}"
                raise ValueError(
                    f"Invalid body reference in joint_child: joint {idx} ('{joint_key}') references child body {joint_child[idx]}, "
                    f"but valid range is [0, {body_count - 1}] (body_count={body_count}). Child cannot be the world (-1)."
                )

            # Validate self-referential joints: parent != child
            self_ref_mask = joint_parent == joint_child
            if np.any(self_ref_mask):
                invalid_indices = np.where(self_ref_mask)[0]
                idx = invalid_indices[0]
                joint_key = self.joint_key[idx] or f"joint_{idx}"
                raise ValueError(
                    f"Self-referential joint: joint {idx} ('{joint_key}') has parent and child both set to body {joint_parent[idx]}."
                )

        # Validate equality constraint body references
        equality_count = len(self.equality_constraint_type)
        if equality_count > 0:
            eq_body1 = np.array(self.equality_constraint_body1, dtype=np.int32)
            invalid_mask = (eq_body1 < -1) | (eq_body1 >= body_count)
            if np.any(invalid_mask):
                invalid_indices = np.where(invalid_mask)[0]
                idx = invalid_indices[0]
                eq_key = self.equality_constraint_key[idx] or f"equality_constraint_{idx}"
                raise ValueError(
                    f"Invalid body reference in equality_constraint_body1: constraint {idx} ('{eq_key}') references body {eq_body1[idx]}, "
                    f"but valid range is [-1, {body_count - 1}] (body_count={body_count})."
                )

            eq_body2 = np.array(self.equality_constraint_body2, dtype=np.int32)
            invalid_mask = (eq_body2 < -1) | (eq_body2 >= body_count)
            if np.any(invalid_mask):
                invalid_indices = np.where(invalid_mask)[0]
                idx = invalid_indices[0]
                eq_key = self.equality_constraint_key[idx] or f"equality_constraint_{idx}"
                raise ValueError(
                    f"Invalid body reference in equality_constraint_body2: constraint {idx} ('{eq_key}') references body {eq_body2[idx]}, "
                    f"but valid range is [-1, {body_count - 1}] (body_count={body_count})."
                )

            # Validate equality constraint joint references
            eq_joint1 = np.array(self.equality_constraint_joint1, dtype=np.int32)
            invalid_mask = (eq_joint1 < -1) | (eq_joint1 >= joint_count)
            if np.any(invalid_mask):
                invalid_indices = np.where(invalid_mask)[0]
                idx = invalid_indices[0]
                eq_key = self.equality_constraint_key[idx] or f"equality_constraint_{idx}"
                raise ValueError(
                    f"Invalid joint reference in equality_constraint_joint1: constraint {idx} ('{eq_key}') references joint {eq_joint1[idx]}, "
                    f"but valid range is [-1, {joint_count - 1}] (joint_count={joint_count})."
                )

            eq_joint2 = np.array(self.equality_constraint_joint2, dtype=np.int32)
            invalid_mask = (eq_joint2 < -1) | (eq_joint2 >= joint_count)
            if np.any(invalid_mask):
                invalid_indices = np.where(invalid_mask)[0]
                idx = invalid_indices[0]
                eq_key = self.equality_constraint_key[idx] or f"equality_constraint_{idx}"
                raise ValueError(
                    f"Invalid joint reference in equality_constraint_joint2: constraint {idx} ('{eq_key}') references joint {eq_joint2[idx]}, "
                    f"but valid range is [-1, {joint_count - 1}] (joint_count={joint_count})."
                )

        # Validate start array monotonicity
        if joint_count > 0:
            joint_q_start = np.array(self.joint_q_start, dtype=np.int32)
            if len(joint_q_start) > 1:
                diffs = np.diff(joint_q_start)
                if np.any(diffs < 0):
                    idx = np.where(diffs < 0)[0][0]
                    raise ValueError(
                        f"joint_q_start is not monotonically increasing: "
                        f"joint_q_start[{idx}]={joint_q_start[idx]} > joint_q_start[{idx + 1}]={joint_q_start[idx + 1]}."
                    )

            joint_qd_start = np.array(self.joint_qd_start, dtype=np.int32)
            if len(joint_qd_start) > 1:
                diffs = np.diff(joint_qd_start)
                if np.any(diffs < 0):
                    idx = np.where(diffs < 0)[0][0]
                    raise ValueError(
                        f"joint_qd_start is not monotonically increasing: "
                        f"joint_qd_start[{idx}]={joint_qd_start[idx]} > joint_qd_start[{idx + 1}]={joint_qd_start[idx + 1]}."
                    )

        articulation_count = self.articulation_count
        if articulation_count > 0:
            articulation_start = np.array(self.articulation_start, dtype=np.int32)
            if len(articulation_start) > 1:
                diffs = np.diff(articulation_start)
                if np.any(diffs < 0):
                    idx = np.where(diffs < 0)[0][0]
                    raise ValueError(
                        f"articulation_start is not monotonically increasing: "
                        f"articulation_start[{idx}]={articulation_start[idx]} > articulation_start[{idx + 1}]={articulation_start[idx + 1]}."
                    )

        # Validate array length consistency
        if joint_count > 0:
            # Per-DOF arrays should have length == joint_dof_count
            dof_arrays = [
                ("joint_axis", self.joint_axis),
                ("joint_armature", self.joint_armature),
                ("joint_target_ke", self.joint_target_ke),
                ("joint_target_kd", self.joint_target_kd),
                ("joint_limit_lower", self.joint_limit_lower),
                ("joint_limit_upper", self.joint_limit_upper),
                ("joint_limit_ke", self.joint_limit_ke),
                ("joint_limit_kd", self.joint_limit_kd),
                ("joint_target_pos", self.joint_target_pos),
                ("joint_target_vel", self.joint_target_vel),
                ("joint_effort_limit", self.joint_effort_limit),
                ("joint_velocity_limit", self.joint_velocity_limit),
                ("joint_friction", self.joint_friction),
                ("joint_act_mode", self.joint_act_mode),
            ]
            for name, arr in dof_arrays:
                if len(arr) != self.joint_dof_count:
                    raise ValueError(
                        f"Array length mismatch: {name} has length {len(arr)}, "
                        f"but expected {self.joint_dof_count} (joint_dof_count)."
                    )

            # Per-coord arrays should have length == joint_coord_count
            coord_arrays = [
                ("joint_q", self.joint_q),
            ]
            for name, arr in coord_arrays:
                if len(arr) != self.joint_coord_count:
                    raise ValueError(
                        f"Array length mismatch: {name} has length {len(arr)}, "
                        f"but expected {self.joint_coord_count} (joint_coord_count)."
                    )

            # Start arrays should have length == joint_count
            start_arrays = [
                ("joint_q_start", self.joint_q_start),
                ("joint_qd_start", self.joint_qd_start),
            ]
            for name, arr in start_arrays:
                if len(arr) != joint_count:
                    raise ValueError(
                        f"Array length mismatch: {name} has length {len(arr)}, "
                        f"but expected {joint_count} (joint_count)."
                    )

    def validate_joint_ordering(self) -> bool:
        """Validate that joints within articulations follow DFS topological ordering.

        This check ensures that joints are ordered such that parent bodies are processed
        before child bodies within each articulation. This ordering is required by some
        solvers (e.g., MuJoCo) for correct kinematic computations.

        This method is public and opt-in because the check has O(n log n) complexity
        due to topological sorting. It is skipped by default in finalize().

        Warns:
            UserWarning: If joints are not in DFS topological order.

        Returns:
            bool: True if joints are correctly ordered, False otherwise.
        """
        from ..utils import topological_sort  # noqa: PLC0415

        if self.joint_count == 0:
            return True

        joint_parent = np.array(self.joint_parent, dtype=np.int32)
        joint_child = np.array(self.joint_child, dtype=np.int32)
        joint_articulation = np.array(self.joint_articulation, dtype=np.int32)

        # Get unique articulations (excluding -1 which means not in any articulation)
        articulation_ids = np.unique(joint_articulation)
        articulation_ids = articulation_ids[articulation_ids >= 0]

        all_ordered = True

        for art_id in articulation_ids:
            # Get joints in this articulation
            art_joints = np.where(joint_articulation == art_id)[0]
            if len(art_joints) <= 1:
                continue

            # Build joint list for topological sort
            joints_simple = [(int(joint_parent[i]), int(joint_child[i])) for i in art_joints]

            try:
                joint_order = topological_sort(joints_simple, use_dfs=True, custom_indices=list(art_joints))

                # Check if current order matches expected DFS order
                if any(joint_order[i] != art_joints[i] for i in range(len(joints_simple))):
                    art_key = (
                        self.articulation_key[art_id]
                        if art_id < len(self.articulation_key)
                        else f"articulation_{art_id}"
                    )
                    warnings.warn(
                        f"Joints in articulation '{art_key}' (id={art_id}) are not in DFS topological order. "
                        f"This may cause issues with some solvers (e.g., MuJoCo). "
                        f"Current order: {list(art_joints)}, expected: {joint_order}.",
                        stacklevel=2,
                    )
                    all_ordered = False
            except ValueError as e:
                # Topological sort failed (e.g., cycle detected)
                art_key = (
                    self.articulation_key[art_id] if art_id < len(self.articulation_key) else f"articulation_{art_id}"
                )
                warnings.warn(
                    f"Failed to validate joint ordering for articulation '{art_key}' (id={art_id}): {e}",
                    stacklevel=2,
                )
                all_ordered = False

        return all_ordered

    def _build_world_starts(self):
        """
        Constructs the per-world entity start indices.

        This method validates that the per-world start index lists for various entities
        (particles, bodies, shapes, joints, articulations, equality constraints and joint
        coordinates/DOFs/constraints) are cumulative and match the total counts of those
        entities. Moreover, it appends the start of tail-end global entities and the
        overall total counts to the end of each start index lists.

        The format of the start index lists is as follows (where `*` can be `body`, `shape`, `joint`, etc.):
            .. code-block:: python

                world_*_start = [ start_world_0, start_world_1, ..., start_world_N , start_global_tail, total_count]

        This allows retrieval of per-world counts using:
            .. code-block:: python

                global_*_count = start_world_0 + (total_count - start_global_tail)
                world_*_count[w] = world_*_start[w + 1] - world_*_start[w]

        e.g.
            .. code-block:: python

                body_world = [-1, -1, 0, 0, ..., 1, 1, ..., N - 1, N - 1, ..., -1, -1, -1, ...]
                body_world_start = [2, 15, 25, ..., 50, 60, 72]
                #          world :  -1 |  0 |  1   ... |  N-1 | -1 |  total
        """
        # List of all world starts of entities
        world_entity_start_arrays = [
            (self.particle_world_start, self.particle_count, self.particle_world, "particle"),
            (self.body_world_start, self.body_count, self.body_world, "body"),
            (self.shape_world_start, self.shape_count, self.shape_world, "shape"),
            (self.joint_world_start, self.joint_count, self.joint_world, "joint"),
            (self.articulation_world_start, self.articulation_count, self.articulation_world, "articulation"),
            (
                self.equality_constraint_world_start,
                len(self.equality_constraint_type),
                self.equality_constraint_world,
                "equality constraint",
            ),
        ]

        def build_entity_start_array(
            entity_count: int, entity_world: list[int], world_entity_start: list[int], name: str
        ):
            # Ensure that entity_world has length equal to entity_count
            if len(entity_world) != entity_count:
                raise ValueError(
                    f"World array for {name}s has incorrect length: expected {entity_count}, found {len(entity_world)}."
                )

            # Initialize world_entity_start with zeros
            world_entity_start.clear()
            world_entity_start.extend([0] * (self.num_worlds + 2))

            # Count global entities at the front of the entity_world array
            front_global_entity_count = 0
            for w in entity_world:
                if w == -1:
                    front_global_entity_count += 1
                else:
                    break
            world_entity_start[0] = front_global_entity_count

            # Compute per-world cumulative counts
            entity_world_np = np.asarray(entity_world, dtype=np.int32)
            world_counts = np.bincount(entity_world_np[entity_world_np >= 0], minlength=self.num_worlds)
            for w in range(self.num_worlds):
                world_entity_start[w + 1] = world_entity_start[w] + int(world_counts[w])

            # Set the last element to the total entity counts over all worlds in the model
            world_entity_start[-1] = entity_count

        # Check that all world offset indices are cumulative and match counts
        for world_start_array, total_count, entity_world_array, name in world_entity_start_arrays:
            # First build the start lists by appending tail-end global and total entity counts
            build_entity_start_array(total_count, entity_world_array, world_start_array, name)

            # Ensure the world_start array has length num_worlds + 2 (for global entities at start/end)
            expected_length = self.num_worlds + 2
            if len(world_start_array) != expected_length:
                raise ValueError(
                    f"World start indices for {name}s have incorrect length: "
                    f"expected {expected_length}, found {len(world_start_array)}."
                )

            # Ensure that per-world start indices are non-decreasing and compute sum of per-world counts
            sum_of_counts = world_start_array[0]
            for w in range(self.num_worlds + 1):
                start_idx = world_start_array[w]
                end_idx = world_start_array[w + 1]
                count = end_idx - start_idx
                if count < 0:
                    raise ValueError(
                        f"Invalid world start indices for {name}s: world {w} has negative count ({count}). "
                        f"Start index: {start_idx}, end index: {end_idx}."
                    )
                sum_of_counts += count

            # Ensure the sum of per-world counts equals the total count
            if sum_of_counts != total_count:
                raise ValueError(
                    f"Sum of per-world {name} counts does not equal total count: "
                    f"expected {total_count}, found {sum_of_counts}."
                )

            # Ensure that the last entry equals the total count
            if world_start_array[-1] != total_count:
                raise ValueError(
                    f"World start indices for {name}s do not match total count: "
                    f"expected final index {total_count}, found {world_start_array[-1]}."
                )

        # List of world starts of joints spaces, i.e. coords/DOFs/constraints
        world_joint_space_start_arrays = [
            (self.joint_dof_world_start, self.joint_qd_start, self.joint_dof_count, "joint DOF"),
            (self.joint_coord_world_start, self.joint_q_start, self.joint_coord_count, "joint coordinate"),
            (self.joint_constraint_world_start, self.joint_cts_start, self.joint_constraint_count, "joint constraint"),
        ]

        def build_joint_space_start_array(
            space_count: int, joint_space_start: list[int], world_space_start: list[int], name: str
        ):
            # Ensure that joint_space_start has length equal to self.joint_count
            if len(joint_space_start) != self.joint_count:
                raise ValueError(
                    f"Joint start array for {name}s has incorrect length: "
                    f"expected {self.joint_count}, found {len(joint_space_start)}."
                )

            # Initialize world_space_start with zeros
            world_space_start.clear()
            world_space_start.extend([0] * (self.num_worlds + 2))

            # Extend joint_space_start with total count to enable computing per-world counts
            joint_space_start_ext = copy.copy(joint_space_start)
            joint_space_start_ext.append(space_count)

            # Count global entities at the front of the entity_world array
            front_global_space_count = 0
            for j, w in enumerate(self.joint_world):
                if w == -1:
                    front_global_space_count += joint_space_start_ext[j + 1] - joint_space_start_ext[j]
                else:
                    break

            # Compute per-world cumulative joint space counts to initialize world_space_start
            for j, w in enumerate(self.joint_world):
                if w >= 0:
                    world_space_start[w + 1] += joint_space_start_ext[j + 1] - joint_space_start_ext[j]

            # Convert per-world counts to cumulative start indices
            world_space_start[0] += front_global_space_count
            for w in range(self.num_worlds):
                world_space_start[w + 1] += world_space_start[w]

            # Add total (i.e. final) entity counts to the per-world start indices
            world_space_start[-1] = space_count

        # Check that all world offset indices are cumulative and match counts
        for world_start_array, space_start_array, total_count, name in world_joint_space_start_arrays:
            # First finalize the start array by appending tail-end global and total entity counts
            build_joint_space_start_array(total_count, space_start_array, world_start_array, name)

            # Ensure the world_start array has length num_worlds + 2 (for global entities at start/end)
            expected_length = self.num_worlds + 2
            if len(world_start_array) != expected_length:
                raise ValueError(
                    f"World start indices for {name}s have incorrect length: "
                    f"expected {expected_length}, found {len(world_start_array)}."
                )

            # Ensure that per-world start indices are non-decreasing and compute sum of per-world counts
            sum_of_counts = world_start_array[0]
            for w in range(self.num_worlds + 1):
                start_idx = world_start_array[w]
                end_idx = world_start_array[w + 1]
                count = end_idx - start_idx
                if count < 0:
                    raise ValueError(
                        f"Invalid world start indices for {name}s: world {w} has negative count ({count}). "
                        f"Start index: {start_idx}, end index: {end_idx}."
                    )
                sum_of_counts += count

            # Ensure the sum of per-world counts equals the total count
            if sum_of_counts != total_count:
                raise ValueError(
                    f"Sum of per-world {name} counts does not equal total count: "
                    f"expected {total_count}, found {sum_of_counts}."
                )

            # Ensure that the last entry equals the total count
            if world_start_array[-1] != total_count:
                raise ValueError(
                    f"World start indices for {name}s do not match total count: "
                    f"expected final index {total_count}, found {world_start_array[-1]}."
                )

    def finalize(
        self,
        device: Devicelike | None = None,
        requires_grad: bool = False,
        skip_all_validations: bool = False,
        skip_validation_worlds: bool = False,
        skip_validation_joints: bool = False,
        skip_validation_shapes: bool = False,
        skip_validation_structure: bool = False,
        skip_validation_joint_ordering: bool = True,
    ) -> Model:
        """
        Finalize the builder and create a concrete :class:`~newton.Model` for simulation.

        This method transfers all simulation data from the builder to device memory,
        returning a Model object ready for simulation. It should be called after all
        elements (particles, bodies, shapes, joints, etc.) have been added to the builder.

        Args:
            device: The simulation device to use (e.g., 'cpu', 'cuda'). If None, uses the current Warp device.
            requires_grad: If True, enables gradient computation for the model (for differentiable simulation).
            skip_all_validations: If True, skips all validation checks. Use for maximum performance when
                you are confident the model is valid. Default is False.
            skip_validation_worlds: If True, skips validation of world ordering and contiguity. Default is False.
            skip_validation_joints: If True, skips validation of joints belonging to an articulation. Default is False.
            skip_validation_shapes: If True, skips validation of shapes having valid contact margins. Default is False.
            skip_validation_structure: If True, skips validation of structural invariants (body/joint references,
                array lengths, monotonicity). Default is False.
            skip_validation_joint_ordering: If True, skips validation of DFS topological joint ordering within
                articulations. Default is True (opt-in) because this check has O(n log n) complexity.

        Returns:
            Model: A fully constructed Model object containing all simulation data on the specified device.

        Notes:
            - Performs validation and correction of rigid body inertia and mass properties.
            - Closes all start-index arrays (e.g., for muscles, joints, articulations) with sentinel values.
            - Sets up all arrays and properties required for simulation, including particles, bodies, shapes,
              joints, springs, muscles, constraints, and collision/contact data.
        """

        # ensure the world count is set correctly
        self.num_worlds = max(1, self.num_worlds)

        # validate world ordering and contiguity
        if not skip_all_validations and not skip_validation_worlds:
            self._validate_world_ordering()

        # validate joints belong to an articulation
        if not skip_all_validations and not skip_validation_joints:
            self._validate_joints()

        # validate shapes have valid contact margins
        if not skip_all_validations and not skip_validation_shapes:
            self._validate_shapes()

        # validate structural invariants (body/joint references, array lengths)
        if not skip_all_validations and not skip_validation_structure:
            self._validate_structure()

        # validate DFS topological joint ordering (opt-in, skipped by default)
        if not skip_all_validations and not skip_validation_joint_ordering:
            self.validate_joint_ordering()

        # construct world starts by ensuring they are cumulative and appending
        # tail-end global counts and sum total counts over the entire model.
        # This method also performs relevant validation checks on the start.
        self._build_world_starts()

        # construct particle inv masses
        ms = np.array(self.particle_mass, dtype=np.float32)
        # static particles (with zero mass) have zero inverse mass
        particle_inv_mass = np.divide(1.0, ms, out=np.zeros_like(ms), where=ms != 0.0)

        with wp.ScopedDevice(device):
            # -------------------------------------
            # construct Model (non-time varying) data

            m = Model(device)
            m.request_contact_attributes(*self._requested_contact_attributes)
            m.request_state_attributes(*self._requested_state_attributes)
            m.requires_grad = requires_grad

            m.num_worlds = self.num_worlds

            # ---------------------
            # particles

            # state (initial)
            m.particle_q = wp.array(self.particle_q, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_qd = wp.array(self.particle_qd, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_mass = wp.array(self.particle_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_inv_mass = wp.array(particle_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_radius = wp.array(self.particle_radius, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_flags = wp.array([flag_to_int(f) for f in self.particle_flags], dtype=wp.int32)
            m.particle_world = wp.array(self.particle_world, dtype=wp.int32)
            m.particle_max_radius = np.max(self.particle_radius) if len(self.particle_radius) > 0 else 0.0
            m.particle_max_velocity = self.particle_max_velocity

            particle_colors = np.empty(self.particle_count, dtype=int)
            for color in range(len(self.particle_color_groups)):
                particle_colors[self.particle_color_groups[color]] = color
            m.particle_colors = wp.array(particle_colors, dtype=int)
            m.particle_color_groups = [wp.array(group, dtype=int) for group in self.particle_color_groups]

            # hash-grid for particle interactions
            if self.particle_count > 1 and m.particle_max_radius > 0.0:
                m.particle_grid = wp.HashGrid(128, 128, 128)
            else:
                m.particle_grid = None

            # ---------------------
            # collision geometry

            m.shape_key = self.shape_key
            m.shape_transform = wp.array(self.shape_transform, dtype=wp.transform, requires_grad=requires_grad)
            m.shape_body = wp.array(self.shape_body, dtype=wp.int32)
            m.shape_flags = wp.array(self.shape_flags, dtype=wp.int32)
            m.body_shapes = self.body_shapes

            # build list of ids for geometry sources (meshes, sdfs)
            geo_sources = []
            finalized_geos = {}  # do not duplicate geometry
            for geo in self.shape_source:
                geo_hash = hash(geo)  # avoid repeated hash computations
                if geo:
                    if geo_hash not in finalized_geos:
                        if isinstance(geo, Mesh):
                            finalized_geos[geo_hash] = geo.finalize(device=device)
                        else:
                            finalized_geos[geo_hash] = geo.finalize()
                    geo_sources.append(finalized_geos[geo_hash])
                else:
                    # add null pointer
                    geo_sources.append(0)

            m.shape_type = wp.array(self.shape_type, dtype=wp.int32)
            m.shape_source_ptr = wp.array(geo_sources, dtype=wp.uint64)
            m.shape_scale = wp.array(self.shape_scale, dtype=wp.vec3, requires_grad=requires_grad)
            m.shape_is_solid = wp.array(self.shape_is_solid, dtype=wp.bool)
            m.shape_thickness = wp.array(self.shape_thickness, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_collision_radius = wp.array(
                self.shape_collision_radius, dtype=wp.float32, requires_grad=requires_grad
            )
            m.shape_world = wp.array(self.shape_world, dtype=wp.int32)

            m.shape_source = self.shape_source  # used for rendering

            m.shape_material_ke = wp.array(self.shape_material_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_kd = wp.array(self.shape_material_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_kf = wp.array(self.shape_material_kf, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_ka = wp.array(self.shape_material_ka, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_mu = wp.array(self.shape_material_mu, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_restitution = wp.array(
                self.shape_material_restitution, dtype=wp.float32, requires_grad=requires_grad
            )
            m.shape_material_torsional_friction = wp.array(
                self.shape_material_torsional_friction, dtype=wp.float32, requires_grad=requires_grad
            )
            m.shape_material_rolling_friction = wp.array(
                self.shape_material_rolling_friction, dtype=wp.float32, requires_grad=requires_grad
            )
            m.shape_material_k_hydro = wp.array(
                self.shape_material_k_hydro, dtype=wp.float32, requires_grad=requires_grad
            )
            m.shape_contact_margin = wp.array(self.shape_contact_margin, dtype=wp.float32, requires_grad=requires_grad)

            m.shape_collision_filter_pairs = {
                (min(s1, s2), max(s1, s2)) for s1, s2 in self.shape_collision_filter_pairs
            }
            m.shape_collision_group = wp.array(self.shape_collision_group, dtype=wp.int32)

            # ---------------------
            # Compute local AABBs and voxel resolutions for contact reduction
            local_aabb_lower = []
            local_aabb_upper = []
            voxel_resolution = []
            voxel_budget = 100  # Maximum voxels per shape for contact reduction

            # Cache per unique (shape_type, shape_params, margin) to avoid redundant AABB computation
            # for instanced shapes (e.g., 256 robots sharing the same shape parameters)
            shape_aabb_cache = {}

            def compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget):
                """Compute voxel resolution from AABB with given budget."""
                size = aabb_upper - aabb_lower
                size = np.maximum(size, 1e-6)  # Avoid division by zero

                # Target voxel size for approximately cubic voxels
                volume = size[0] * size[1] * size[2]
                v = (volume / voxel_budget) ** (1.0 / 3.0)
                v = max(v, 1e-6)

                # Initial resolution
                nx = max(1, round(size[0] / v))
                ny = max(1, round(size[1] / v))
                nz = max(1, round(size[2] / v))

                # Reduce until under budget (reduce largest axis first for more cubic voxels)
                while nx * ny * nz > voxel_budget:
                    if nx >= ny and nx >= nz and nx > 1:
                        nx -= 1
                    elif ny >= nz and ny > 1:
                        ny -= 1
                    elif nz > 1:
                        nz -= 1
                    else:
                        break

                return nx, ny, nz

            for shape_idx, (shape_type, shape_src, shape_scale) in enumerate(
                zip(self.shape_type, self.shape_source, self.shape_scale, strict=True)
            ):
                # Get margin to expand AABB (SDF extends beyond shape bounds by margin + thickness)
                margin = self.shape_contact_margin[shape_idx] + self.shape_thickness[shape_idx]

                # Create cache key based on shape type and parameters
                if (shape_type == GeoType.MESH or shape_type == GeoType.CONVEX_MESH) and shape_src is not None:
                    cache_key = (shape_type, id(shape_src), tuple(shape_scale), margin)
                else:
                    cache_key = (shape_type, tuple(shape_scale), margin)

                # Check cache first
                if cache_key in shape_aabb_cache:
                    aabb_lower, aabb_upper, nx, ny, nz = shape_aabb_cache[cache_key]
                else:
                    # Compute AABB based on shape type
                    if shape_type == GeoType.MESH and shape_src is not None:
                        # Compute local AABB from mesh vertices
                        vertices = shape_src.vertices
                        aabb_lower = vertices.min(axis=0)
                        aabb_upper = vertices.max(axis=0)

                        # Apply scale to get the actual local-space bounds
                        aabb_lower = aabb_lower * np.array(shape_scale)
                        aabb_upper = aabb_upper * np.array(shape_scale)

                        # Expand by margin (SDF extends beyond mesh bounds)
                        aabb_lower = aabb_lower - margin
                        aabb_upper = aabb_upper + margin

                        nx, ny, nz = compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget)

                    elif shape_type == GeoType.CONVEX_MESH and shape_src is not None:
                        # Compute local AABB from convex mesh vertices (similar to MESH)
                        vertices = shape_src.vertices
                        aabb_lower = vertices.min(axis=0)
                        aabb_upper = vertices.max(axis=0)

                        # Apply scale to get the actual local-space bounds
                        aabb_lower = aabb_lower * np.array(shape_scale)
                        aabb_upper = aabb_upper * np.array(shape_scale)

                        # Expand by margin
                        aabb_lower = aabb_lower - margin
                        aabb_upper = aabb_upper + margin

                        nx, ny, nz = compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget)

                    elif shape_type == GeoType.ELLIPSOID:
                        # Ellipsoid: shape_scale = (semi_axis_x, semi_axis_y, semi_axis_z)
                        sx, sy, sz = shape_scale
                        aabb_lower = np.array([-sx - margin, -sy - margin, -sz - margin])
                        aabb_upper = np.array([sx + margin, sy + margin, sz + margin])
                        nx, ny, nz = compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget)

                    elif shape_type == GeoType.BOX:
                        # Box: shape_scale = (hx, hy, hz) half-extents
                        hx, hy, hz = shape_scale
                        aabb_lower = np.array([-hx - margin, -hy - margin, -hz - margin])
                        aabb_upper = np.array([hx + margin, hy + margin, hz + margin])
                        nx, ny, nz = compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget)

                    elif shape_type == GeoType.SPHERE:
                        # Sphere: shape_scale = (radius, radius, radius)
                        r = shape_scale[0] + margin
                        aabb_lower = np.array([-r, -r, -r])
                        aabb_upper = np.array([r, r, r])
                        nx, ny, nz = compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget)

                    elif shape_type == GeoType.CAPSULE:
                        # Capsule: shape_scale = (radius, half_height, radius)
                        # Capsule is along Z axis with hemispherical caps (matches SDF in kernels.py)
                        r, half_height, _ = shape_scale
                        r_expanded = r + margin
                        aabb_lower = np.array([-r_expanded, -r_expanded, -half_height - r_expanded])
                        aabb_upper = np.array([r_expanded, r_expanded, half_height + r_expanded])
                        nx, ny, nz = compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget)

                    elif shape_type == GeoType.CYLINDER:
                        # Cylinder: shape_scale = (radius, half_height, radius)
                        # Cylinder is along Z axis (matches SDF in kernels.py)
                        r, half_height, _ = shape_scale
                        aabb_lower = np.array([-r - margin, -r - margin, -half_height - margin])
                        aabb_upper = np.array([r + margin, r + margin, half_height + margin])
                        nx, ny, nz = compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget)

                    elif shape_type == GeoType.CONE:
                        # Cone: shape_scale = (radius, half_height, radius)
                        # Cone is along Z axis (matches SDF in kernels.py)
                        r, half_height, _ = shape_scale
                        aabb_lower = np.array([-r - margin, -r - margin, -half_height - margin])
                        aabb_upper = np.array([r + margin, r + margin, half_height + margin])
                        nx, ny, nz = compute_voxel_resolution_from_aabb(aabb_lower, aabb_upper, voxel_budget)

                    else:
                        # Other shapes (PLANE, HFIELD, etc.): use default unit cube with 1x1x1 voxel grid
                        aabb_lower = np.array([-1.0, -1.0, -1.0])
                        aabb_upper = np.array([1.0, 1.0, 1.0])
                        nx, ny, nz = 1, 1, 1

                    # Cache the result for reuse by identical shapes
                    shape_aabb_cache[cache_key] = (aabb_lower, aabb_upper, nx, ny, nz)

                local_aabb_lower.append(aabb_lower)
                local_aabb_upper.append(aabb_upper)
                voxel_resolution.append([nx, ny, nz])

            m.shape_local_aabb_lower = wp.array(local_aabb_lower, dtype=wp.vec3, device=device)
            m.shape_local_aabb_upper = wp.array(local_aabb_upper, dtype=wp.vec3, device=device)
            m.shape_voxel_resolution = wp.array(voxel_resolution, dtype=wp.vec3i, device=device)

            # ---------------------
            # Compute SDFs for mesh shapes (per-shape opt-in via sdf_max_resolution, sdf_target_voxel_size or is_hydroelastic)
            from ..geometry.sdf_utils import (  # noqa: PLC0415
                SDFData,
                compute_sdf,
                create_empty_sdf_data,
            )

            # Check if we're running on GPU - wp.Volume only supports CUDA
            current_device = wp.get_device(device)
            is_gpu = current_device.is_cuda

            # Check if there are any mesh shapes with collision enabled that request SDF generation
            has_sdf_meshes = any(
                stype == GeoType.MESH
                and ssrc is not None
                and sflags & ShapeFlags.COLLIDE_SHAPES
                and (sdf_max_resolution is not None or sdf_target_voxel_size is not None)
                for stype, ssrc, sflags, sdf_max_resolution, sdf_target_voxel_size in zip(
                    self.shape_type,
                    self.shape_source,
                    self.shape_flags,
                    self.shape_sdf_max_resolution,
                    self.shape_sdf_target_voxel_size,
                    strict=True,
                )
            )

            # Check if there are any shapes with hydroelastic collision enabled
            has_hydroelastic_shapes = any(
                (sflags & ShapeFlags.HYDROELASTIC) and (sflags & ShapeFlags.COLLIDE_SHAPES)
                for sflags in self.shape_flags
            )

            if has_sdf_meshes and not is_gpu:
                raise ValueError(
                    "SDF generation for mesh shapes (sdf_max_resolution != None) requires a CUDA-capable GPU device. "
                    "wp.Volume (used for SDF generation) only supports CUDA. "
                    "Either set sdf_max_resolution=None for all mesh shapes or use a CUDA device."
                )

            if has_hydroelastic_shapes and not is_gpu:
                raise ValueError(
                    "Hydroelastic collision (is_hydroelastic=True) requires a CUDA-capable GPU device. "
                    "wp.Volume (used for SDF generation) only supports CUDA. "
                    "Either set is_hydroelastic=False for all shapes or use a CUDA device."
                )

            if has_sdf_meshes or has_hydroelastic_shapes:
                sdf_data_list = []
                # Keep volume objects alive for reference counting
                sdf_volumes = []
                sdf_coarse_volumes = []

                # caches
                sdf_cache = {}

                sdf_block_coords = []  # flat array of coordinates of active SDF tiles
                sdf_shape2blocks = []  # array indexing into sdf_block_coords for each shape. Multiple shapes can index into the same block range.

                # Create empty SDF data once for reuse by non-SDF shapes
                empty_sdf_data = create_empty_sdf_data()

                for i in range(len(self.shape_type)):
                    shape_type = self.shape_type[i]
                    shape_src = self.shape_source[i]
                    shape_flags = self.shape_flags[i]
                    shape_scale = self.shape_scale[i]
                    shape_thickness = self.shape_thickness[i]
                    shape_contact_margin = self.shape_contact_margin[i]
                    sdf_narrow_band_range = self.shape_sdf_narrow_band_range[i]
                    sdf_target_voxel_size = self.shape_sdf_target_voxel_size[i]
                    sdf_max_resolution = self.shape_sdf_max_resolution[i]
                    is_hydroelastic = bool(shape_flags & ShapeFlags.HYDROELASTIC)

                    # Determine if this shape needs SDF:
                    # - Mesh shapes with sdf_max_resolution/sdf_target_voxel_size set, OR
                    # - Any colliding shape with is_hydroelastic=True
                    needs_sdf = (
                        shape_type == GeoType.MESH
                        and shape_src is not None
                        and shape_flags & ShapeFlags.COLLIDE_SHAPES
                        and (sdf_max_resolution is not None or sdf_target_voxel_size is not None)
                    ) or (is_hydroelastic and shape_flags & ShapeFlags.COLLIDE_SHAPES)

                    if needs_sdf:
                        # Mesh-sdf collisions handle shape scaling at collision time,
                        # in which case we can compute SDF for this mesh shape in unscaled local space here.
                        # For hydrelastic collisions this impact of this approximation has yet to be quantified
                        # so we will bake scale into the SDF data here for now.
                        bake_scale = is_hydroelastic

                        cache_key = (
                            hash(shape_src),
                            shape_type,
                            shape_thickness,
                            shape_contact_margin,
                            tuple(sdf_narrow_band_range),
                            sdf_target_voxel_size,
                            sdf_max_resolution,
                            tuple(shape_scale) if bake_scale else None,
                        )
                        if cache_key in sdf_cache:
                            idx = sdf_cache[cache_key]
                            sdf_data = sdf_data_list[idx]
                            sparse_volume = sdf_volumes[idx]
                            coarse_volume = sdf_coarse_volumes[idx]
                            shape2blocks = [sdf_shape2blocks[idx][0], sdf_shape2blocks[idx][1]]
                        else:
                            sdf_data, sparse_volume, coarse_volume, block_coords = compute_sdf(
                                mesh_src=shape_src,
                                shape_type=shape_type,
                                shape_scale=shape_scale,
                                shape_thickness=shape_thickness,
                                narrow_band_distance=sdf_narrow_band_range,
                                margin=shape_contact_margin,
                                target_voxel_size=sdf_target_voxel_size,
                                max_resolution=sdf_max_resolution,
                                bake_scale=bake_scale,
                            )
                            sdf_cache[cache_key] = i
                            block_start_idx = len(sdf_block_coords)
                            num_blocks = len(block_coords)
                            shape2blocks = [block_start_idx, block_start_idx + num_blocks]
                            sdf_block_coords.extend(block_coords)
                    else:
                        # Non-SDF shapes get empty SDFData
                        sdf_data = empty_sdf_data
                        sparse_volume = None
                        coarse_volume = None
                        shape2blocks = [0, 0]

                    sdf_data_list.append(sdf_data)
                    sdf_volumes.append(sparse_volume)
                    sdf_coarse_volumes.append(coarse_volume)
                    sdf_shape2blocks.append(shape2blocks)

                # Create array of SDFData structs
                m.shape_sdf_data = wp.array(sdf_data_list, dtype=SDFData, device=device)
                # Keep volume objects alive for reference counting
                m.shape_sdf_volume = sdf_volumes
                m.shape_sdf_coarse_volume = sdf_coarse_volumes
                m.shape_sdf_block_coords = wp.array(sdf_block_coords, dtype=wp.vec3us)
                m.shape_sdf_shape2blocks = wp.array(sdf_shape2blocks, dtype=wp.vec2i)
            else:
                # SDF mesh-mesh collision and hydroelastics not enabled or no colliding meshes/shapes
                # Still need one SDFData per shape (all empty) so narrow phase can safely access shape_sdf_data[shape_idx]
                empty_sdf_data = create_empty_sdf_data()
                m.shape_sdf_data = wp.array([empty_sdf_data] * len(self.shape_type), dtype=SDFData, device=device)
                m.shape_sdf_volume = [None] * len(self.shape_type)
                m.shape_sdf_coarse_volume = [None] * len(self.shape_type)
                m.shape_sdf_block_coords = wp.array([], dtype=wp.vec3us)
                m.shape_sdf_shape2blocks = wp.array([], dtype=wp.vec2i)

            # ---------------------
            # springs

            def _to_wp_array(data, dtype, requires_grad):
                if len(data) == 0:
                    return None
                return wp.array(data, dtype=dtype, requires_grad=requires_grad)

            m.spring_indices = _to_wp_array(self.spring_indices, wp.int32, requires_grad=False)
            m.spring_rest_length = _to_wp_array(self.spring_rest_length, wp.float32, requires_grad=requires_grad)
            m.spring_stiffness = _to_wp_array(self.spring_stiffness, wp.float32, requires_grad=requires_grad)
            m.spring_damping = _to_wp_array(self.spring_damping, wp.float32, requires_grad=requires_grad)
            m.spring_control = _to_wp_array(self.spring_control, wp.float32, requires_grad=requires_grad)

            # ---------------------
            # triangles

            m.tri_indices = _to_wp_array(self.tri_indices, wp.int32, requires_grad=False)
            m.tri_poses = _to_wp_array(self.tri_poses, wp.mat22, requires_grad=requires_grad)
            m.tri_activations = _to_wp_array(self.tri_activations, wp.float32, requires_grad=requires_grad)
            m.tri_materials = _to_wp_array(self.tri_materials, wp.float32, requires_grad=requires_grad)
            m.tri_areas = _to_wp_array(self.tri_areas, wp.float32, requires_grad=requires_grad)

            # ---------------------
            # edges

            m.edge_indices = _to_wp_array(self.edge_indices, wp.int32, requires_grad=False)
            m.edge_rest_angle = _to_wp_array(self.edge_rest_angle, wp.float32, requires_grad=requires_grad)
            m.edge_rest_length = _to_wp_array(self.edge_rest_length, wp.float32, requires_grad=requires_grad)
            m.edge_bending_properties = _to_wp_array(
                self.edge_bending_properties, wp.float32, requires_grad=requires_grad
            )

            # ---------------------
            # tetrahedra

            m.tet_indices = _to_wp_array(self.tet_indices, wp.int32, requires_grad=False)
            m.tet_poses = _to_wp_array(self.tet_poses, wp.mat33, requires_grad=requires_grad)
            m.tet_activations = _to_wp_array(self.tet_activations, wp.float32, requires_grad=requires_grad)
            m.tet_materials = _to_wp_array(self.tet_materials, wp.float32, requires_grad=requires_grad)

            # -----------------------
            # muscles

            # close the muscle waypoint indices
            muscle_start = copy.copy(self.muscle_start)
            muscle_start.append(len(self.muscle_bodies))

            m.muscle_start = wp.array(muscle_start, dtype=wp.int32)
            m.muscle_params = wp.array(self.muscle_params, dtype=wp.float32, requires_grad=requires_grad)
            m.muscle_bodies = wp.array(self.muscle_bodies, dtype=wp.int32)
            m.muscle_points = wp.array(self.muscle_points, dtype=wp.vec3, requires_grad=requires_grad)
            m.muscle_activations = wp.array(self.muscle_activations, dtype=wp.float32, requires_grad=requires_grad)

            # --------------------------------------
            # rigid bodies

            # Apply inertia verification and correction
            # This catches negative masses/inertias and other critical issues
            if len(self.body_mass) > 0:
                if self.validate_inertia_detailed:
                    # Use detailed Python validation with per-body warnings
                    for i in range(len(self.body_mass)):
                        mass = self.body_mass[i]
                        inertia = self.body_inertia[i]
                        body_key = self.body_key[i] if i < len(self.body_key) else f"body_{i}"

                        corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(
                            mass, inertia, self.balance_inertia, self.bound_mass, self.bound_inertia, body_key
                        )

                        if was_corrected:
                            self.body_mass[i] = corrected_mass
                            self.body_inertia[i] = corrected_inertia
                            # Update inverse mass and inertia
                            if corrected_mass > 0.0:
                                self.body_inv_mass[i] = 1.0 / corrected_mass
                            else:
                                self.body_inv_mass[i] = 0.0

                            if any(x for x in corrected_inertia):
                                self.body_inv_inertia[i] = wp.inverse(corrected_inertia)
                            else:
                                self.body_inv_inertia[i] = corrected_inertia

                    # For detailed validation, create arrays from builder data (which were updated)
                    m.body_mass = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
                    m.body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
                    m.body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
                    m.body_inv_inertia = wp.array(self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad)
                else:
                    # Use fast Warp kernel validation
                    # First create arrays for the kernel
                    body_mass_array = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
                    body_inertia_array = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
                    body_inv_mass_array = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
                    body_inv_inertia_array = wp.array(
                        self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad
                    )
                    correction_flags = wp.zeros(len(self.body_mass), dtype=wp.bool)

                    # Launch validation kernel
                    wp.launch(
                        kernel=validate_and_correct_inertia_kernel,
                        dim=len(self.body_mass),
                        inputs=[
                            body_mass_array,
                            body_inertia_array,
                            body_inv_mass_array,
                            body_inv_inertia_array,
                            self.balance_inertia,
                            self.bound_mass if self.bound_mass is not None else 0.0,
                            self.bound_inertia if self.bound_inertia is not None else 0.0,
                            correction_flags,
                        ],
                    )

                    # Check if any corrections were made
                    num_corrections = int(np.sum(correction_flags.numpy()))
                    if num_corrections > 0:
                        warnings.warn(
                            f"Inertia validation corrected {num_corrections} bodies. "
                            f"Set validate_inertia_detailed=True for detailed per-body warnings.",
                            stacklevel=2,
                        )

                    # Directly use the corrected arrays on the Model (avoids double allocation)
                    # Note: This means the ModelBuilder's internal state is NOT updated for the fast path
                    m.body_mass = body_mass_array
                    m.body_inv_mass = body_inv_mass_array
                    m.body_inertia = body_inertia_array
                    m.body_inv_inertia = body_inv_inertia_array
            else:
                # No bodies, create empty arrays
                m.body_mass = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
                m.body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
                m.body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
                m.body_inv_inertia = wp.array(self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad)

            m.body_q = wp.array(self.body_q, dtype=wp.transform, requires_grad=requires_grad)
            m.body_qd = wp.array(self.body_qd, dtype=wp.spatial_vector, requires_grad=requires_grad)
            m.body_com = wp.array(self.body_com, dtype=wp.vec3, requires_grad=requires_grad)
            m.body_key = self.body_key
            m.body_world = wp.array(self.body_world, dtype=wp.int32)

            # body colors
            if self.body_color_groups:
                body_colors = np.empty(self.body_count, dtype=int)
                for color in range(len(self.body_color_groups)):
                    body_colors[self.body_color_groups[color]] = color
                m.body_colors = wp.array(body_colors, dtype=int)
                m.body_color_groups = [wp.array(group, dtype=int) for group in self.body_color_groups]

            # joints
            m.joint_type = wp.array(self.joint_type, dtype=wp.int32)
            m.joint_parent = wp.array(self.joint_parent, dtype=wp.int32)
            m.joint_child = wp.array(self.joint_child, dtype=wp.int32)
            m.joint_X_p = wp.array(self.joint_X_p, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_X_c = wp.array(self.joint_X_c, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_dof_dim = wp.array(np.array(self.joint_dof_dim), dtype=wp.int32, ndim=2)
            m.joint_axis = wp.array(self.joint_axis, dtype=wp.vec3, requires_grad=requires_grad)
            m.joint_q = wp.array(self.joint_q, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_qd = wp.array(self.joint_qd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_key = self.joint_key
            m.joint_world = wp.array(self.joint_world, dtype=wp.int32)
            # compute joint ancestors
            child_to_joint = {}
            for i, child in enumerate(self.joint_child):
                child_to_joint[child] = i
            parent_joint = []
            for parent in self.joint_parent:
                parent_joint.append(child_to_joint.get(parent, -1))
            m.joint_ancestor = wp.array(parent_joint, dtype=wp.int32)
            m.joint_articulation = wp.array(self.joint_articulation, dtype=wp.int32)

            # dynamics properties
            m.joint_armature = wp.array(self.joint_armature, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_act_mode = wp.array(self.joint_act_mode, dtype=wp.int32)
            m.joint_target_ke = wp.array(self.joint_target_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_kd = wp.array(self.joint_target_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_pos = wp.array(self.joint_target_pos, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_vel = wp.array(self.joint_target_vel, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_f = wp.array(self.joint_f, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_effort_limit = wp.array(self.joint_effort_limit, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_velocity_limit = wp.array(self.joint_velocity_limit, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_friction = wp.array(self.joint_friction, dtype=wp.float32, requires_grad=requires_grad)

            m.joint_limit_lower = wp.array(self.joint_limit_lower, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_upper = wp.array(self.joint_limit_upper, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_ke = wp.array(self.joint_limit_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_kd = wp.array(self.joint_limit_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_enabled = wp.array(self.joint_enabled, dtype=wp.bool)

            # 'close' the start index arrays with a sentinel value
            joint_q_start = copy.copy(self.joint_q_start)
            joint_q_start.append(self.joint_coord_count)
            joint_qd_start = copy.copy(self.joint_qd_start)
            joint_qd_start.append(self.joint_dof_count)
            articulation_start = copy.copy(self.articulation_start)
            articulation_start.append(self.joint_count)

            # Compute max joints and dofs per articulation for IK/Jacobian kernel launches
            max_joints_per_articulation = 0
            max_dofs_per_articulation = 0
            for art_idx in range(len(self.articulation_start)):
                joint_start = articulation_start[art_idx]
                joint_end = articulation_start[art_idx + 1]
                num_joints = joint_end - joint_start
                max_joints_per_articulation = max(max_joints_per_articulation, num_joints)
                # Compute dofs for this articulation
                dof_start = joint_qd_start[joint_start]
                dof_end = joint_qd_start[joint_end]
                num_dofs = dof_end - dof_start
                max_dofs_per_articulation = max(max_dofs_per_articulation, num_dofs)

            m.joint_q_start = wp.array(joint_q_start, dtype=wp.int32)
            m.joint_qd_start = wp.array(joint_qd_start, dtype=wp.int32)
            m.articulation_start = wp.array(articulation_start, dtype=wp.int32)
            m.articulation_key = self.articulation_key
            m.articulation_world = wp.array(self.articulation_world, dtype=wp.int32)
            m.max_joints_per_articulation = max_joints_per_articulation
            m.max_dofs_per_articulation = max_dofs_per_articulation

            # ---------------------
            # equality constraints
            m.equality_constraint_type = wp.array(self.equality_constraint_type, dtype=wp.int32)
            m.equality_constraint_body1 = wp.array(self.equality_constraint_body1, dtype=wp.int32)
            m.equality_constraint_body2 = wp.array(self.equality_constraint_body2, dtype=wp.int32)
            m.equality_constraint_anchor = wp.array(self.equality_constraint_anchor, dtype=wp.vec3)
            m.equality_constraint_torquescale = wp.array(self.equality_constraint_torquescale, dtype=wp.float32)
            m.equality_constraint_relpose = wp.array(
                self.equality_constraint_relpose, dtype=wp.transform, requires_grad=requires_grad
            )
            m.equality_constraint_joint1 = wp.array(self.equality_constraint_joint1, dtype=wp.int32)
            m.equality_constraint_joint2 = wp.array(self.equality_constraint_joint2, dtype=wp.int32)
            m.equality_constraint_polycoef = wp.array(self.equality_constraint_polycoef, dtype=wp.float32)
            m.equality_constraint_key = self.equality_constraint_key
            m.equality_constraint_enabled = wp.array(self.equality_constraint_enabled, dtype=wp.bool)
            m.equality_constraint_world = wp.array(self.equality_constraint_world, dtype=wp.int32)

            # mimic constraints
            m.constraint_mimic_joint0 = wp.array(self.constraint_mimic_joint0, dtype=wp.int32)
            m.constraint_mimic_joint1 = wp.array(self.constraint_mimic_joint1, dtype=wp.int32)
            m.constraint_mimic_coef0 = wp.array(self.constraint_mimic_coef0, dtype=wp.float32)
            m.constraint_mimic_coef1 = wp.array(self.constraint_mimic_coef1, dtype=wp.float32)
            m.constraint_mimic_enabled = wp.array(self.constraint_mimic_enabled, dtype=wp.bool)
            m.constraint_mimic_key = self.constraint_mimic_key
            m.constraint_mimic_world = wp.array(self.constraint_mimic_world, dtype=wp.int32)

            # ---------------------
            # per-world start indices
            m.particle_world_start = wp.array(self.particle_world_start, dtype=wp.int32)
            m.body_world_start = wp.array(self.body_world_start, dtype=wp.int32)
            m.shape_world_start = wp.array(self.shape_world_start, dtype=wp.int32)
            m.joint_world_start = wp.array(self.joint_world_start, dtype=wp.int32)
            m.articulation_world_start = wp.array(self.articulation_world_start, dtype=wp.int32)
            m.equality_constraint_world_start = wp.array(self.equality_constraint_world_start, dtype=wp.int32)
            m.joint_dof_world_start = wp.array(self.joint_dof_world_start, dtype=wp.int32)
            m.joint_coord_world_start = wp.array(self.joint_coord_world_start, dtype=wp.int32)
            m.joint_constraint_world_start = wp.array(self.joint_constraint_world_start, dtype=wp.int32)

            # ---------------------
            # counts
            m.joint_count = self.joint_count
            m.joint_dof_count = self.joint_dof_count
            m.joint_coord_count = self.joint_coord_count
            m.joint_constraint_count = self.joint_constraint_count
            m.particle_count = len(self.particle_q)
            m.body_count = self.body_count
            m.shape_count = len(self.shape_type)
            m.tri_count = len(self.tri_poses)
            m.tet_count = len(self.tet_poses)
            m.edge_count = len(self.edge_rest_angle)
            m.spring_count = len(self.spring_rest_length)
            m.muscle_count = len(self.muscle_start)
            m.articulation_count = len(self.articulation_start)
            m.equality_constraint_count = len(self.equality_constraint_type)
            m.constraint_mimic_count = len(self.constraint_mimic_joint0)

            self.find_shape_contact_pairs(m)

            # enable ground plane
            m.up_axis = self.up_axis

            # set gravity - create per-world gravity array for multi-world support
            if self.world_gravity:
                # Use per-world gravity from world_gravity list
                gravity_vecs = [wp.vec3(*g) for g in self.world_gravity]
            else:
                # Fallback: use scalar gravity for all worlds
                gravity_vec = wp.vec3(*(g * self.gravity for g in self.up_vector))
                gravity_vecs = [gravity_vec] * self.num_worlds
            m.gravity = wp.array(
                gravity_vecs,
                dtype=wp.vec3,
                device=device,
                requires_grad=requires_grad,
            )

            # Add custom attributes onto the model (with lazy evaluation)
            # Early return if no custom attributes exist to avoid overhead
            if not self.custom_attributes:
                return m

            # Resolve authoritative counts for custom frequencies
            # Use incremental _custom_frequency_counts as primary source, with safety fallback
            custom_frequency_counts: dict[str, int] = {}
            frequency_max_lens: dict[str, int] = {}  # Track max len(values) per frequency as fallback

            # First pass: collect max len(values) per frequency as fallback
            for _full_key, custom_attr in self.custom_attributes.items():
                freq_key = custom_attr.frequency_key
                if isinstance(freq_key, str):
                    attr_len = len(custom_attr.values) if custom_attr.values else 0
                    frequency_max_lens[freq_key] = max(frequency_max_lens.get(freq_key, 0), attr_len)

            # Determine authoritative counts: prefer _custom_frequency_counts, fallback to max lens
            for freq_key, max_len in frequency_max_lens.items():
                if freq_key in self._custom_frequency_counts:
                    # Use authoritative incremental counter
                    custom_frequency_counts[freq_key] = self._custom_frequency_counts[freq_key]
                else:
                    # Safety fallback: use max observed length
                    custom_frequency_counts[freq_key] = max_len

            # Relaxed validation: warn about attributes with fewer values than frequency count
            for full_key, custom_attr in self.custom_attributes.items():
                freq_key = custom_attr.frequency_key
                if isinstance(freq_key, str):
                    attr_count = len(custom_attr.values) if custom_attr.values else 0
                    expected_count = custom_frequency_counts[freq_key]
                    if attr_count < expected_count:
                        warnings.warn(
                            f"Custom attribute '{full_key}' has {attr_count} values but frequency '{freq_key}' "
                            f"expects {expected_count}. Missing values will be filled with defaults.",
                            UserWarning,
                            stacklevel=2,
                        )

            # Store custom frequency counts on the model for selection.py and other consumers
            m.custom_frequency_counts = custom_frequency_counts

            # Process custom attributes
            for _full_key, custom_attr in self.custom_attributes.items():
                freq_key = custom_attr.frequency_key

                # determine count by frequency
                if isinstance(freq_key, str):
                    # Custom frequency: count determined by validated frequency count
                    count = custom_frequency_counts.get(freq_key, 0)
                elif freq_key == Model.AttributeFrequency.ONCE:
                    count = 1
                elif freq_key == Model.AttributeFrequency.BODY:
                    count = m.body_count
                elif freq_key == Model.AttributeFrequency.SHAPE:
                    count = m.shape_count
                elif freq_key == Model.AttributeFrequency.JOINT:
                    count = m.joint_count
                elif freq_key == Model.AttributeFrequency.JOINT_DOF:
                    count = m.joint_dof_count
                elif freq_key == Model.AttributeFrequency.JOINT_COORD:
                    count = m.joint_coord_count
                elif freq_key == Model.AttributeFrequency.JOINT_CONSTRAINT:
                    count = m.joint_constraint_count
                elif freq_key == Model.AttributeFrequency.ARTICULATION:
                    count = m.articulation_count
                elif freq_key == Model.AttributeFrequency.WORLD:
                    count = m.num_worlds
                elif freq_key == Model.AttributeFrequency.EQUALITY_CONSTRAINT:
                    count = m.equality_constraint_count
                elif freq_key == Model.AttributeFrequency.CONSTRAINT_MIMIC:
                    count = m.constraint_mimic_count
                elif freq_key == Model.AttributeFrequency.PARTICLE:
                    count = m.particle_count
                elif freq_key == Model.AttributeFrequency.EDGE:
                    count = m.edge_count
                elif freq_key == Model.AttributeFrequency.TRIANGLE:
                    count = m.tri_count
                elif freq_key == Model.AttributeFrequency.TETRAHEDRON:
                    count = m.tet_count
                elif freq_key == Model.AttributeFrequency.SPRING:
                    count = m.spring_count
                else:
                    continue

                # Skip empty custom frequency attributes
                if count == 0:
                    continue

                result = custom_attr.build_array(count, device=device, requires_grad=requires_grad)
                m.add_attribute(custom_attr.name, result, freq_key, custom_attr.assignment, custom_attr.namespace)

            return m

    def _test_group_pair(self, group_a: int, group_b: int) -> bool:
        """Test if two collision groups should interact.

        This matches the exact logic from broad_phase_common.test_group_pair kernel function.

        Args:
            group_a: First collision group ID
            group_b: Second collision group ID

        Returns:
            bool: True if the groups should collide, False otherwise
        """
        if group_a == 0 or group_b == 0:
            return False
        if group_a > 0:
            return group_a == group_b or group_b < 0
        if group_a < 0:
            return group_a != group_b
        return False

    def _test_world_and_group_pair(
        self, world_a: int, world_b: int, collision_group_a: int, collision_group_b: int
    ) -> bool:
        """Test if two entities should collide based on world indices and collision groups.

        This matches the exact logic from broad_phase_common.test_world_and_group_pair kernel function.

        Args:
            world_a: World index of first entity
            world_b: World index of second entity
            collision_group_a: Collision group of first entity
            collision_group_b: Collision group of second entity

        Returns:
            bool: True if the entities should collide, False otherwise
        """
        # Check world indices first
        if world_a != -1 and world_b != -1 and world_a != world_b:
            return False

        # If same world or at least one is global (-1), check collision groups
        return self._test_group_pair(collision_group_a, collision_group_b)

    def find_shape_contact_pairs(self, model: Model):
        """
        Identifies and stores all potential shape contact pairs for collision detection.

        This method examines the collision groups and collision masks of all shapes in the model
        to determine which pairs of shapes should be considered for contact generation. It respects
        any user-specified collision filter pairs to avoid redundant or undesired contacts.

        The resulting contact pairs are stored in the model as a 2D array of shape indices.

        Uses the exact same filtering logic as the broad phase kernels (test_world_and_group_pair)
        to ensure consistency between EXPLICIT mode (precomputed pairs) and NXN/SAP modes.

        Args:
            model (Model): The simulation model to which the contact pairs will be assigned.

        Side Effects:
            - Sets `model.shape_contact_pairs` to a wp.array of shape pairs (wp.vec2i).
            - Sets `model.shape_contact_pair_count` to the number of contact pairs found.
        """
        filters: set[tuple[int, int]] = model.shape_collision_filter_pairs
        contact_pairs: list[tuple[int, int]] = []

        # Keep only colliding shapes (those with COLLIDE_SHAPES flag) and sort by world for optimization
        colliding_indices = [i for i, flag in enumerate(self.shape_flags) if flag & ShapeFlags.COLLIDE_SHAPES]
        sorted_indices = sorted(colliding_indices, key=lambda i: self.shape_world[i])

        # Iterate over all pairs of colliding shapes
        for i1 in range(len(sorted_indices)):
            s1 = sorted_indices[i1]
            world1 = self.shape_world[s1]
            collision_group1 = self.shape_collision_group[s1]

            for i2 in range(i1 + 1, len(sorted_indices)):
                s2 = sorted_indices[i2]
                world2 = self.shape_world[s2]
                collision_group2 = self.shape_collision_group[s2]

                # Early break optimization: if both shapes are in non-global worlds and different worlds,
                # they can never collide. Since shapes are sorted by world, all remaining shapes will also
                # be in different worlds, so we can break early.
                if world1 != -1 and world2 != -1 and world1 != world2:
                    break

                # Apply the exact same filtering logic as test_world_and_group_pair kernel
                if not self._test_world_and_group_pair(world1, world2, collision_group1, collision_group2):
                    continue

                if s1 > s2:
                    shape_a, shape_b = s2, s1
                else:
                    shape_a, shape_b = s1, s2

                # Skip if explicitly filtered
                if (shape_a, shape_b) not in filters:
                    contact_pairs.append((shape_a, shape_b))

        model.shape_contact_pairs = wp.array(np.array(contact_pairs), dtype=wp.vec2i, device=model.device)
        model.shape_contact_pair_count = len(contact_pairs)
