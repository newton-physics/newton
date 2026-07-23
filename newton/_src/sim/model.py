# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implementation of the Newton model class."""

from __future__ import annotations

import logging
import operator
import warnings
from collections.abc import Callable, Iterable, Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, SupportsIndex

import numpy as np
import warp as wp

from ..core.types import Devicelike, override
from ..utils.mesh import MeshAdjacency, MeshAdjacencyData
from .contacts import Contacts
from .control import Control
from .state import State

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..actuators.actuator import Actuator
    from ..utils.heightfield import HeightfieldData
    from .collide import CollisionPipeline


_HAS_HEIGHTFIELDS_DEPRECATION_MSG = (
    "Model.has_heightfields is deprecated; use Model.heightfield_count, "
    "or model.heightfield_count > 0 for boolean checks, instead."
)

_SHAPE_COLLISION_FILTER_MUTATION_DEPRECATION_MSG = (
    "Mutating Model.shape_collision_filter_pairs after ModelBuilder.finalize() is deprecated. "
    "Configure collision filters on ModelBuilder before finalizing; post-finalize filter changes "
    "do not rebuild Model.shape_contact_pairs."
)


def _pack_shape_pair_codes(shape_a: np.ndarray, shape_b: np.ndarray) -> np.ndarray:
    """Pack shape index pairs into ``int64`` codes ordered like canonical tuples.

    Args:
        shape_a: First shape indices, shape [pair_count].
        shape_b: Second shape indices, shape [pair_count].

    Returns:
        ``(min << 32) | max`` codes, shape [pair_count]. Sorting these codes
        sorts the canonical ``(min, max)`` pairs lexicographically.
    """
    lo = np.minimum(shape_a, shape_b).astype(np.int64)
    hi = np.maximum(shape_a, shape_b)
    return (lo << 32) | hi


def _unpack_shape_pair_codes(codes: np.ndarray) -> np.ndarray:
    """Unpack ``int64`` pair codes into canonical pairs, shape [pair_count, 2]."""
    pairs = np.empty((codes.shape[0], 2), dtype=np.int32)
    pairs[:, 0] = codes >> 32
    pairs[:, 1] = codes & 0xFFFFFFFF
    return pairs


class _DeprecatedShapeCollisionFilterSet(set[tuple[int, int]]):
    """Mutation-deprecated compat view over the canonical filter-pair array.

    The canonical store is ``packed``: a sorted, unique 1-D ``int64`` array of
    pair codes (see :func:`_pack_shape_pair_codes`). The public
    :attr:`Model.shape_collision_filter_pairs` descriptor materializes this
    view into native set contents on access, so plain ``set`` reads work
    unchanged; internal consumers use :meth:`contains_pair`,
    :meth:`mask_pairs`, and :meth:`pairs_array`, which query the packed array
    while it exists and transparently fall back to native set contents after a
    deprecated mutation drops it.
    """

    __hash__ = None

    def __init__(self, pairs: Iterable[tuple[int, int]] = (), packed: np.ndarray | None = None):
        super().__init__((shape_a, shape_b) if shape_a <= shape_b else (shape_b, shape_a) for shape_a, shape_b in pairs)
        self._packed = packed
        self._pairs_array: np.ndarray | None = None
        self._materialized = packed is None

    @staticmethod
    def _canonical_pair(shape_a: int, shape_b: int) -> tuple[int, int]:
        return (shape_a, shape_b) if shape_a <= shape_b else (shape_b, shape_a)

    @classmethod
    def _canonical_pair_from_object(cls, pair: object) -> tuple[int, int] | None:
        if not isinstance(pair, tuple) or len(pair) != 2:
            return None
        shape_a, shape_b = pair
        return cls._canonical_pair(shape_a, shape_b)

    @classmethod
    def _iter_canonical_pairs(cls, pairs: Iterable[object]) -> Iterator[tuple[int, int]]:
        for pair in pairs:
            canonical_pair = cls._canonical_pair_from_object(pair)
            if canonical_pair is not None:
                yield canonical_pair

    def _ensure_materialized(self) -> None:
        if not self._materialized:
            if self._packed is not None and self._packed.shape[0] > 0:
                super().update(map(tuple, _unpack_shape_pair_codes(self._packed).tolist()))
            self._materialized = True

    @property
    def is_materialized(self) -> bool:
        return self._materialized

    def materialize(self) -> None:
        self._ensure_materialized()

    def packed_pairs(self) -> np.ndarray | None:
        """Sorted unique packed pair codes, or ``None`` after mutation."""
        return self._packed

    def contains_pair(self, shape_a: SupportsIndex, shape_b: SupportsIndex) -> bool:
        """Return membership of a shape pair in any argument order."""
        # Normalize to Python ints: NumPy int32 inputs would overflow the
        # 32-bit shift in the packed code and give wrong membership.
        shape_a, shape_b = operator.index(shape_a), operator.index(shape_b)
        if shape_a > shape_b:
            shape_a, shape_b = shape_b, shape_a
        if self._packed is None:
            return super().__contains__((shape_a, shape_b))
        code = (shape_a << 32) | shape_b
        index = int(np.searchsorted(self._packed, code))
        return bool(index < self._packed.shape[0] and self._packed[index] == code)

    def mask_pairs(self, pairs: np.ndarray) -> np.ndarray:
        """Return a boolean membership mask for shape pairs in any order."""
        if pairs.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        if self._packed is None:
            return np.fromiter(
                (self.contains_pair(shape_a, shape_b) for shape_a, shape_b in pairs),
                dtype=bool,
                count=pairs.shape[0],
            )
        if self._packed.shape[0] == 0:
            return np.zeros(pairs.shape[0], dtype=bool)
        codes = _pack_shape_pair_codes(pairs[:, 0], pairs[:, 1])
        index = np.searchsorted(self._packed, codes)
        in_range = index < self._packed.shape[0]
        return in_range & (self._packed[np.minimum(index, self._packed.shape[0] - 1)] == codes)

    def pairs_array(self) -> np.ndarray:
        """Canonical pairs sorted lexicographically, shape [pair_count, 2].

        The returned array is read-only: while the packed store exists it
        aliases the cached canonical pairs, and mutating it would corrupt
        every later filter query and the materialized public set.
        """
        if self._packed is None:
            pairs = np.asarray(sorted(self), dtype=np.int32).reshape((-1, 2))
        else:
            if self._pairs_array is None:
                self._pairs_array = _unpack_shape_pair_codes(self._packed)
                self._pairs_array.setflags(write=False)
            return self._pairs_array
        pairs.setflags(write=False)
        return pairs

    def _prepare_mutation(self) -> None:
        self._ensure_materialized()
        self._packed = None
        self._pairs_array = None
        warnings.warn(_SHAPE_COLLISION_FILTER_MUTATION_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)

    def __bool__(self) -> bool:
        if self._packed is not None and self._packed.shape[0] > 0:
            return True
        return super().__len__() != 0

    # Generic consumers (viewer-file serialization, deepcopy) iterate the raw
    # __dict__ value without the materializing descriptor; keep iteration and
    # length lazy-safe so they never observe a half-empty set.
    @override
    def __iter__(self) -> Iterator[tuple[int, int]]:
        self._ensure_materialized()
        return super().__iter__()

    @override
    def __len__(self) -> int:
        self._ensure_materialized()
        return super().__len__()

    @override
    def add(self, element: tuple[int, int]) -> None:
        self._prepare_mutation()
        shape_a, shape_b = element
        super().add(self._canonical_pair(shape_a, shape_b))

    @override
    def clear(self) -> None:
        self._prepare_mutation()
        super().clear()

    @override
    def discard(self, element: object) -> None:
        self._prepare_mutation()
        canonical_pair = self._canonical_pair_from_object(element)
        if canonical_pair is not None:
            super().discard(canonical_pair)

    @override
    def pop(self) -> tuple[int, int]:
        self._prepare_mutation()
        return super().pop()

    @override
    def remove(self, element: tuple[int, int]) -> None:
        self._prepare_mutation()
        shape_a, shape_b = element
        super().remove(self._canonical_pair(shape_a, shape_b))

    @override
    def update(self, *others: Iterable[tuple[int, int]]) -> None:
        self._prepare_mutation()
        super().update(self._canonical_pair(shape_a, shape_b) for other in others for shape_a, shape_b in other)

    @override
    def difference_update(self, *others: Iterable[object]) -> None:
        self._prepare_mutation()
        for other in others:
            for canonical_pair in self._iter_canonical_pairs(other):
                super().discard(canonical_pair)

    @override
    def intersection_update(self, *others: Iterable[object]) -> None:
        self._prepare_mutation()
        canonical_others = [set(self._iter_canonical_pairs(other)) for other in others]
        super().intersection_update(*canonical_others)

    @override
    def symmetric_difference_update(self, other: Iterable[tuple[int, int]]) -> None:
        self._prepare_mutation()
        super().symmetric_difference_update(self._canonical_pair(shape_a, shape_b) for shape_a, shape_b in other)

    @override
    def __ior__(self, other: Iterable[tuple[int, int]]):
        self._prepare_mutation()
        super().update(self._canonical_pair(shape_a, shape_b) for shape_a, shape_b in other)
        return self

    @override
    def __iand__(self, other: AbstractSet[object]):
        self._prepare_mutation()
        super().intersection_update(set(self._iter_canonical_pairs(other)))
        return self

    @override
    def __isub__(self, other: AbstractSet[object]):
        self._prepare_mutation()
        super().difference_update(set(self._iter_canonical_pairs(other)))
        return self

    @override
    def __ixor__(self, other: Iterable[tuple[int, int]]):
        self._prepare_mutation()
        super().symmetric_difference_update({self._canonical_pair(shape_a, shape_b) for shape_a, shape_b in other})
        return self


class _ShapeCollisionFilterPairsAttribute:
    """Set of canonical shape index pairs that should not collide.

    Mutating or reassigning this finalized-model set is deprecated. Configure
    collision filters on :class:`ModelBuilder` before calling
    :meth:`ModelBuilder.finalize` instead; post-finalize changes do not rebuild
    :attr:`Model.shape_contact_pairs`.
    """

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        if instance is None:
            return self
        filters = instance.__dict__.get("shape_collision_filter_pairs")
        if filters is None:
            filters = _DeprecatedShapeCollisionFilterSet()
            instance.__dict__["shape_collision_filter_pairs"] = filters
        if isinstance(filters, _DeprecatedShapeCollisionFilterSet):
            filters.materialize()
        return filters

    def __set__(self, instance: Any, value: Iterable[tuple[int, int]]) -> None:
        if instance.__dict__.get("shape_collision_filter_pairs") is value:
            return
        if "shape_collision_filter_pairs" in instance.__dict__:
            warnings.warn(_SHAPE_COLLISION_FILTER_MUTATION_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        instance.__dict__["shape_collision_filter_pairs"] = _DeprecatedShapeCollisionFilterSet(value)


class Model:
    """
    Represents the static (non-time-varying) definition of a simulation model in Newton.

    The Model class encapsulates all geometry, constraints, and parameters that describe a physical system
    for simulation. It is designed to be constructed via the ModelBuilder, which handles the correct
    initialization and population of all fields.

    Key Features:
        - Stores all static data for simulation: particles, rigid bodies, joints, shapes, soft/rigid elements, etc.
        - Supports grouping of entities by world using world indices (e.g., `particle_world`, `body_world`, etc.).
          - Index -1: global entities shared across all worlds.
          - Indices 0, 1, 2, ...: world-specific entities.
        - Grouping enables:
          - Collision detection optimization (e.g., separating worlds)
          - Visualization (e.g., spatially separating worlds)
          - Parallel processing of independent worlds

    Note:
        It is strongly recommended to use the :class:`ModelBuilder` to construct a Model.
        Direct instantiation and manual population of Model fields is possible but discouraged.
    """

    if TYPE_CHECKING:
        shape_collision_filter_pairs: set[tuple[int, int]]
    else:
        shape_collision_filter_pairs = _ShapeCollisionFilterPairsAttribute()

    class AttributeAssignment(IntEnum):
        """Enumeration of attribute assignment categories.

        Defines which component of the simulation system owns and manages specific attributes.
        This categorization determines where custom attributes are attached during simulation
        object creation (Model, State, Control, or Contacts).
        """

        MODEL = 0
        """Model attributes are attached to the :class:`~newton.Model` object."""
        STATE = 1
        """State attributes are attached to the :class:`~newton.State` object."""
        CONTROL = 2
        """Control attributes are attached to the :class:`~newton.Control` object."""
        CONTACT = 3
        """Contact attributes are attached to the :class:`~newton.Contacts` object."""

    class AttributeFrequency(IntEnum):
        """Enumeration of attribute frequency categories.

        Defines the dimensional structure and indexing pattern for custom attributes.
        This determines how many elements an attribute array should have and how it
        should be indexed in relation to the model's entities such as joints, bodies, shapes, etc.
        """

        ONCE = 0
        """Attribute frequency is a single value."""
        JOINT = 1
        """Attribute frequency follows the number of joints (see :attr:`~newton.Model.joint_count`)."""
        JOINT_DOF = 2
        """Attribute frequency follows the number of joint degrees of freedom (see :attr:`~newton.Model.joint_dof_count`)."""
        JOINT_COORD = 3
        """Attribute frequency follows the number of joint positional coordinates (see :attr:`~newton.Model.joint_coord_count`)."""
        JOINT_CONSTRAINT = 4
        """Attribute frequency follows the number of joint constraints (see :attr:`~newton.Model.joint_constraint_count`)."""
        BODY = 5
        """Attribute frequency follows the number of bodies (see :attr:`~newton.Model.body_count`)."""
        SHAPE = 6
        """Attribute frequency follows the number of shapes (see :attr:`~newton.Model.shape_count`)."""
        ARTICULATION = 7
        """Attribute frequency follows the number of articulations (see :attr:`~newton.Model.articulation_count`)."""
        PARTICLE = 9
        """Attribute frequency follows the number of particles (see :attr:`~newton.Model.particle_count`)."""
        EDGE = 10
        """Attribute frequency follows the number of edges (see :attr:`~newton.Model.edge_count`)."""
        TRIANGLE = 11
        """Attribute frequency follows the number of triangles (see :attr:`~newton.Model.tri_count`)."""
        TETRAHEDRON = 12
        """Attribute frequency follows the number of tetrahedra (see :attr:`~newton.Model.tet_count`)."""
        SPRING = 13
        """Attribute frequency follows the number of springs (see :attr:`~newton.Model.spring_count`)."""
        CONSTRAINT_MIMIC = 14
        """Attribute frequency follows the number of mimic constraints (see :attr:`~newton.Model.constraint_mimic_count`)."""
        WORLD = 15
        """Attribute frequency follows the number of worlds (see :attr:`~newton.Model.world_count`)."""

    @dataclass(frozen=True)
    class AttributeSpec:
        """Semantic metadata for an indexed model attribute.

        .. experimental::

            ``compaction_policy`` is part of the experimental coupled-solver
            framework and may change without a deprecation period.
        """

        frequency: Model.AttributeFrequency | str
        """Entity domain that determines the attribute row count."""
        assignment: Model.AttributeAssignment | None = None
        """Object that owns the attribute, or ``None`` when it belongs to the :class:`Model`."""
        references: Model.AttributeFrequency | str | None = None
        """Entity domain referenced by integer values, or ``None`` when values are not entity indices."""
        row_width: int = 1
        """Number of flattened values stored for each entity row."""
        requires_empty_sentinel: bool = False
        """Whether empty compacted storage retains a sentinel value."""
        deprecated: bool = False
        """Whether this is a deprecated compatibility alias that generic consumers should skip."""
        alias_of: str | None = None
        """Canonical name used when explicitly accessing this alias through a model view."""
        compaction_policy: Literal["generic", "end", "start", "world_start", "color_groups", "passthrough"] = "generic"
        """Experimental policy used by coupled model views.

        ``"generic"`` selects and remaps rows using this spec; ``"end"``
        remaps exclusive boundaries in the referenced domain; ``"start"``,
        ``"world_start"``, and ``"color_groups"`` select their corresponding
        structured handling; and ``"passthrough"`` disables automatic count
        limiting. Non-generic policies may still be overridden by the coupled
        solver when constructing a compact view.
        """

        def __post_init__(self) -> None:
            if self.row_width < 1:
                raise ValueError(f"Attribute row_width must be positive, got {self.row_width}")
            if self.compaction_policy not in {
                "generic",
                "end",
                "start",
                "world_start",
                "color_groups",
                "passthrough",
            }:
                raise ValueError(f"Unknown attribute compaction policy {self.compaction_policy!r}")
            if self.compaction_policy == "end" and self.references is None:
                raise ValueError("Attribute compaction policy 'end' requires a reference domain")

    _CORE_ATTRIBUTE_SPECS: ClassVar[dict[str, AttributeSpec]] = {
        # particles
        "particle_q": AttributeSpec(AttributeFrequency.PARTICLE),
        "particle_qd": AttributeSpec(AttributeFrequency.PARTICLE),
        "particle_mass": AttributeSpec(AttributeFrequency.PARTICLE),
        "particle_inv_mass": AttributeSpec(AttributeFrequency.PARTICLE),
        "particle_radius": AttributeSpec(AttributeFrequency.PARTICLE),
        "particle_flags": AttributeSpec(AttributeFrequency.PARTICLE),
        "particle_world": AttributeSpec(AttributeFrequency.PARTICLE, references=AttributeFrequency.WORLD),
        "particle_colors": AttributeSpec(AttributeFrequency.PARTICLE),
        "particle_world_start": AttributeSpec(
            AttributeFrequency.PARTICLE,
            compaction_policy="world_start",
        ),
        "particle_color_groups": AttributeSpec(
            AttributeFrequency.PARTICLE,
            compaction_policy="color_groups",
        ),
        # bodies
        "body_q": AttributeSpec(AttributeFrequency.BODY),
        "body_qd": AttributeSpec(AttributeFrequency.BODY),
        "body_com": AttributeSpec(AttributeFrequency.BODY),
        "body_inertia": AttributeSpec(AttributeFrequency.BODY),
        "body_inv_inertia": AttributeSpec(AttributeFrequency.BODY),
        "body_mass": AttributeSpec(AttributeFrequency.BODY),
        "body_inv_mass": AttributeSpec(AttributeFrequency.BODY),
        "body_flags": AttributeSpec(AttributeFrequency.BODY),
        "body_f": AttributeSpec(AttributeFrequency.BODY),
        "body_label": AttributeSpec(AttributeFrequency.BODY),
        "body_world": AttributeSpec(AttributeFrequency.BODY, references=AttributeFrequency.WORLD),
        "body_colors": AttributeSpec(AttributeFrequency.BODY),
        "body_world_start": AttributeSpec(
            AttributeFrequency.BODY,
            compaction_policy="world_start",
        ),
        "body_color_groups": AttributeSpec(
            AttributeFrequency.BODY,
            compaction_policy="color_groups",
        ),
        "body_shapes": AttributeSpec(
            AttributeFrequency.ONCE,
            compaction_policy="passthrough",
        ),
        # shapes
        "shape_label": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_transform": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_body": AttributeSpec(
            AttributeFrequency.SHAPE,
            references=AttributeFrequency.BODY,
        ),
        "shape_flags": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_ke": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_kd": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_kf": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_ka": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_mu": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_restitution": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_mu_torsional": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_mu_rolling": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_material_kh": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_gap": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_type": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_is_solid": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_margin": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_source": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_source_ptr": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_scale": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_color": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_filter": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_collision_group": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_collision_radius": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_world": AttributeSpec(AttributeFrequency.SHAPE, references=AttributeFrequency.WORLD),
        "shape_heightfield_index": AttributeSpec(
            AttributeFrequency.SHAPE,
            requires_empty_sentinel=True,
        ),
        "shape_edge_range": AttributeSpec(AttributeFrequency.SHAPE, requires_empty_sentinel=True),
        "_shape_sdf_index": AttributeSpec(AttributeFrequency.SHAPE),
        "_shape_mesh_properties": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_collision_aabb_lower": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_collision_aabb_upper": AttributeSpec(AttributeFrequency.SHAPE),
        "_shape_voxel_resolution": AttributeSpec(AttributeFrequency.SHAPE),
        "shape_world_start": AttributeSpec(
            AttributeFrequency.SHAPE,
            compaction_policy="world_start",
        ),
        "shape_collision_filter_pairs": AttributeSpec(
            AttributeFrequency.ONCE,
            compaction_policy="passthrough",
        ),
        "shape_contact_pairs": AttributeSpec(
            AttributeFrequency.ONCE,
            compaction_policy="passthrough",
        ),
        # springs and finite elements
        "spring_indices": AttributeSpec(
            AttributeFrequency.SPRING,
            references=AttributeFrequency.PARTICLE,
            row_width=2,
        ),
        "spring_rest_length": AttributeSpec(AttributeFrequency.SPRING),
        "spring_stiffness": AttributeSpec(AttributeFrequency.SPRING),
        "spring_damping": AttributeSpec(AttributeFrequency.SPRING),
        "spring_control": AttributeSpec(AttributeFrequency.SPRING),
        "spring_constraint_lambdas": AttributeSpec(AttributeFrequency.SPRING),
        "tri_indices": AttributeSpec(
            AttributeFrequency.TRIANGLE,
            references=AttributeFrequency.PARTICLE,
        ),
        "tri_poses": AttributeSpec(AttributeFrequency.TRIANGLE),
        "tri_activations": AttributeSpec(AttributeFrequency.TRIANGLE),
        "tri_materials": AttributeSpec(AttributeFrequency.TRIANGLE),
        "tri_areas": AttributeSpec(AttributeFrequency.TRIANGLE),
        "edge_indices": AttributeSpec(
            AttributeFrequency.EDGE,
            references=AttributeFrequency.PARTICLE,
        ),
        "edge_rest_angle": AttributeSpec(AttributeFrequency.EDGE),
        "edge_rest_length": AttributeSpec(AttributeFrequency.EDGE),
        "edge_bending_properties": AttributeSpec(AttributeFrequency.EDGE),
        "edge_constraint_lambdas": AttributeSpec(AttributeFrequency.EDGE),
        "tet_indices": AttributeSpec(
            AttributeFrequency.TETRAHEDRON,
            references=AttributeFrequency.PARTICLE,
        ),
        "tet_poses": AttributeSpec(AttributeFrequency.TETRAHEDRON),
        "tet_activations": AttributeSpec(AttributeFrequency.TETRAHEDRON),
        "tet_materials": AttributeSpec(AttributeFrequency.TETRAHEDRON),
        # joints
        "joint_type": AttributeSpec(AttributeFrequency.JOINT),
        "joint_parent": AttributeSpec(AttributeFrequency.JOINT, references=AttributeFrequency.BODY),
        "joint_child": AttributeSpec(AttributeFrequency.JOINT, references=AttributeFrequency.BODY),
        "joint_ancestor": AttributeSpec(
            AttributeFrequency.JOINT,
            references=AttributeFrequency.JOINT,
        ),
        "joint_articulation": AttributeSpec(
            AttributeFrequency.JOINT,
            references=AttributeFrequency.ARTICULATION,
        ),
        "joint_X_p": AttributeSpec(AttributeFrequency.JOINT),
        "joint_X_c": AttributeSpec(AttributeFrequency.JOINT),
        "joint_dof_dim": AttributeSpec(AttributeFrequency.JOINT),
        "joint_enabled": AttributeSpec(AttributeFrequency.JOINT),
        "joint_twist_lower": AttributeSpec(AttributeFrequency.JOINT),
        "joint_twist_upper": AttributeSpec(AttributeFrequency.JOINT),
        "joint_label": AttributeSpec(AttributeFrequency.JOINT),
        "joint_world": AttributeSpec(AttributeFrequency.JOINT, references=AttributeFrequency.WORLD),
        "joint_q_start": AttributeSpec(
            AttributeFrequency.JOINT,
            references=AttributeFrequency.JOINT_COORD,
            compaction_policy="start",
        ),
        "joint_qd_start": AttributeSpec(
            AttributeFrequency.JOINT,
            references=AttributeFrequency.JOINT_DOF,
            compaction_policy="start",
        ),
        "joint_world_start": AttributeSpec(
            AttributeFrequency.JOINT,
            compaction_policy="world_start",
        ),
        "joint_dof_world_start": AttributeSpec(
            AttributeFrequency.JOINT_DOF,
            compaction_policy="world_start",
        ),
        "joint_coord_world_start": AttributeSpec(
            AttributeFrequency.JOINT_COORD,
            compaction_policy="world_start",
        ),
        "joint_constraint_world_start": AttributeSpec(
            AttributeFrequency.JOINT_CONSTRAINT,
            compaction_policy="world_start",
        ),
        "joint_q": AttributeSpec(AttributeFrequency.JOINT_COORD),
        "joint_qd": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_f": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_armature": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_target_qd": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_act": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_axis": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_target_mode": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_target_ke": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_target_kd": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_damping": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_limit_lower": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_limit_upper": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_limit_ke": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_limit_kd": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_effort_limit": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_friction": AttributeSpec(AttributeFrequency.JOINT_DOF),
        "joint_velocity_limit": AttributeSpec(AttributeFrequency.JOINT_DOF),
        # articulations and mimic constraints
        "articulation_start": AttributeSpec(
            AttributeFrequency.ARTICULATION,
            references=AttributeFrequency.JOINT,
            compaction_policy="start",
        ),
        "articulation_end": AttributeSpec(
            AttributeFrequency.ARTICULATION,
            references=AttributeFrequency.JOINT,
            compaction_policy="end",
        ),
        "articulation_label": AttributeSpec(AttributeFrequency.ARTICULATION),
        "articulation_world": AttributeSpec(
            AttributeFrequency.ARTICULATION,
            references=AttributeFrequency.WORLD,
        ),
        "articulation_world_start": AttributeSpec(
            AttributeFrequency.ARTICULATION,
            compaction_policy="world_start",
        ),
        "constraint_mimic_joint0": AttributeSpec(
            AttributeFrequency.CONSTRAINT_MIMIC,
            references=AttributeFrequency.JOINT,
        ),
        "constraint_mimic_joint1": AttributeSpec(
            AttributeFrequency.CONSTRAINT_MIMIC,
            references=AttributeFrequency.JOINT,
        ),
        "constraint_mimic_coef0": AttributeSpec(AttributeFrequency.CONSTRAINT_MIMIC),
        "constraint_mimic_coef1": AttributeSpec(AttributeFrequency.CONSTRAINT_MIMIC),
        "constraint_mimic_enabled": AttributeSpec(AttributeFrequency.CONSTRAINT_MIMIC),
        "constraint_mimic_label": AttributeSpec(AttributeFrequency.CONSTRAINT_MIMIC),
        "constraint_mimic_world": AttributeSpec(
            AttributeFrequency.CONSTRAINT_MIMIC,
            references=AttributeFrequency.WORLD,
        ),
    }

    _ATTRIBUTE_FREQUENCY_COUNT_ATTRS: ClassVar[dict[AttributeFrequency, str]] = {
        AttributeFrequency.JOINT: "joint_count",
        AttributeFrequency.JOINT_DOF: "joint_dof_count",
        AttributeFrequency.JOINT_COORD: "joint_coord_count",
        AttributeFrequency.JOINT_CONSTRAINT: "joint_constraint_count",
        AttributeFrequency.BODY: "body_count",
        AttributeFrequency.SHAPE: "shape_count",
        AttributeFrequency.ARTICULATION: "articulation_count",
        AttributeFrequency.PARTICLE: "particle_count",
        AttributeFrequency.EDGE: "edge_count",
        AttributeFrequency.TRIANGLE: "tri_count",
        AttributeFrequency.TETRAHEDRON: "tet_count",
        AttributeFrequency.SPRING: "spring_count",
        AttributeFrequency.CONSTRAINT_MIMIC: "constraint_mimic_count",
        AttributeFrequency.WORLD: "world_count",
    }

    class AttributeNamespace:
        """
        A container for namespaced custom attributes.

        Custom attributes are stored as regular instance attributes on this object,
        allowing hierarchical organization of related properties.
        """

        def __init__(self, name: str):
            """Initialize the namespace container.

            Args:
                name: The name of the namespace
            """
            object.__setattr__(self, "_name", name)
            object.__setattr__(self, "_deprecated_aliases", {})

        def add_deprecated_alias(self, name: str, getter: Callable[[], Any], message: str) -> None:
            """Add a deprecated attribute alias.

            Args:
                name: Alias name exposed on the namespace.
                getter: Callable returning the canonical target object.
                message: Deprecation warning message.
            """
            if name in self.__dict__ or name in self._deprecated_aliases:
                raise AttributeError(f"Attribute already exists: {self._name}.{name}")
            self._deprecated_aliases[name] = (getter, message)

        def __getattr__(self, name: str) -> Any:
            aliases = self.__dict__.get("_deprecated_aliases", {})
            if name in aliases:
                getter, message = aliases[name]
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return getter()
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def __setattr__(self, name: str, value: Any) -> None:
            if not name.startswith("_"):
                aliases = object.__getattribute__(self, "__dict__").get("_deprecated_aliases", {})
                if name in aliases:
                    getter, message = aliases[name]
                    warnings.warn(message, DeprecationWarning, stacklevel=2)
                    target = getter()
                    if isinstance(target, wp.array):
                        target.assign(value)
                        return
                    raise AttributeError(f"Deprecated alias '{self._name}.{name}' does not support assignment")
            object.__setattr__(self, name, value)

        def __repr__(self):
            """Return a string representation showing the namespace and its attributes."""
            # List all public attributes (not starting with _)
            attrs = [k for k in self.__dict__ if not k.startswith("_")]
            attrs.extend(k for k in self._deprecated_aliases if k not in attrs)
            return f"AttributeNamespace('{self._name}', attributes={attrs})"

    def __init__(self, device: Devicelike | None = None):
        """
        Initialize a Model object.

        Args:
            device: Device on which the Model's data will be allocated.
        """
        self.requires_grad: bool = False
        """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""
        self.world_count: int = 0
        """Number of worlds added to the ModelBuilder."""

        self.particle_q: wp.array[wp.vec3] | None = None
        """Particle positions [m], shape [particle_count, 3], float."""
        self.particle_qd: wp.array[wp.vec3] | None = None
        """Particle velocities [m/s], shape [particle_count, 3], float."""
        self.particle_mass: wp.array[wp.float32] | None = None
        """Particle mass [kg], shape [particle_count], float."""
        self.particle_inv_mass: wp.array[wp.float32] | None = None
        """Particle inverse mass [1/kg], shape [particle_count], float."""
        self.particle_radius: wp.array[wp.float32] | None = None
        """Particle radius [m], shape [particle_count], float."""
        self.particle_max_radius: float = 0.0
        """Maximum particle radius [m] (useful for HashGrid construction)."""
        self.particle_ke: float = 1.0e3
        """Particle normal contact stiffness [N/m] (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_kd: float = 1.0e2
        """Particle normal contact damping [N·s/m] (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_kf: float = 1.0e2
        """Particle contact friction gain [N·s/m] (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_mu: float = 0.5
        """Particle friction coefficient [dimensionless]."""
        self.particle_cohesion: float = 0.0
        """Particle cohesion strength [m]."""
        self.particle_adhesion: float = 0.0
        """Particle adhesion strength [m]."""
        self.particle_grid: wp.HashGrid | None = None
        """HashGrid instance for accelerated simulation of particle interactions."""
        self.particle_flags: wp.array[wp.int32] | None = None
        """Particle enabled state, shape [particle_count], int."""
        self.particle_max_velocity: float = 1e5
        """Maximum particle velocity [m/s] (to prevent instability)."""
        self.particle_world: wp.array[wp.int32] | None = None
        """World index for each particle, shape [particle_count], int. -1 for global."""
        self.particle_world_start: wp.array[wp.int32] | None = None
        """Start index of the first particle per world, shape [world_count + 2], int.

        The entries at indices ``0`` to ``world_count - 1`` store the start index of
        the particles belonging to that world. The second-last element (accessible
        via index ``-2``) stores the start index of the global particles (i.e. with
        world index ``-1``) added to the end of the model, and the last element
        stores the total particle count.

        The number of particles in a given world ``w`` can be computed as::

            num_particles_in_world = particle_world_start[w + 1] - particle_world_start[w]

        The total number of global particles can be computed as::

            num_global_particles = particle_world_start[-1] - particle_world_start[-2] + particle_world_start[0]
        """

        self.shape_label: list[str] = []
        """List of labels for each shape."""
        self.shape_transform: wp.array[wp.transform] | None = None
        """Rigid shape transforms [m, unitless quaternion], shape [shape_count, 7], float."""
        self.shape_body: wp.array[wp.int32] | None = None
        """Rigid shape body index, shape [shape_count], int."""
        self.shape_flags: wp.array[wp.int32] | None = None
        """Rigid shape flags, shape [shape_count], int."""
        self.body_shapes: dict[int, list[int]] = {}
        """Mapping from body index to list of attached shape indices."""

        # Shape material properties
        self.shape_material_ke: wp.array[wp.float32] | None = None
        """Shape contact elastic stiffness [N/m], shape [shape_count], float."""
        self.shape_material_kd: wp.array[wp.float32] | None = None
        """Shape contact damping [N·s/m], shape [shape_count], float."""
        self.shape_material_kf: wp.array[wp.float32] | None = None
        """Shape contact friction gain [N·s/m], shape [shape_count], float."""
        self.shape_material_ka: wp.array[wp.float32] | None = None
        """Shape contact adhesion distance [m], shape [shape_count], float."""
        self.shape_material_mu: wp.array[wp.float32] | None = None
        """Shape coefficient of friction [dimensionless], shape [shape_count], float."""
        self.shape_material_restitution: wp.array[wp.float32] | None = None
        """Shape coefficient of restitution [dimensionless], shape [shape_count], float."""
        self.shape_material_mu_torsional: wp.array[wp.float32] | None = None
        """Shape torsional friction coefficient [dimensionless] (resistance to spinning at contact point), shape [shape_count], float."""
        self.shape_material_mu_rolling: wp.array[wp.float32] | None = None
        """Shape rolling friction coefficient [dimensionless] (resistance to rolling motion), shape [shape_count], float."""
        self.shape_material_kh: wp.array[wp.float32] | None = None
        """Shape hydroelastic stiffness coefficient [N/m^3], shape [shape_count], float.
        Under the default linear pressure law, contact force scales with
        contact area, ``kh``, and penetration depth."""
        self.shape_gap: wp.array[wp.float32] | None = None
        """Shape additional contact detection gap [m], shape [shape_count], float."""

        # Shape geometry properties
        self.shape_type: wp.array[wp.int32] | None = None
        """Shape geometry type, shape [shape_count], int32."""
        self.shape_is_solid: wp.array[wp.bool] | None = None
        """Whether shape is solid or hollow, shape [shape_count], bool."""
        self.shape_margin: wp.array[wp.float32] | None = None
        """Shape surface margin [m], shape [shape_count], float."""
        self.shape_source: list[object | None] = []
        """List of source geometry objects (e.g., :class:`~newton.Mesh`) used for broadphase collision detection and rendering, shape [shape_count]."""
        self.shape_source_ptr: wp.array[wp.uint64] | None = None
        """Geometry source pointers to be used inside the Warp kernels which are generated by finalizing the geometry objects, see for example :meth:`newton.Mesh.finalize`, shape [shape_count], uint64."""
        self.shape_scale: wp.array[wp.vec3] | None = None
        """Shape 3D scale, shape [shape_count], vec3."""
        self.shape_color: wp.array[wp.vec3] | None = None
        """Shape display colors [0, 1], shape [shape_count], vec3."""
        self.shape_filter: wp.array[wp.int32] | None = None
        """Shape filter group, shape [shape_count], int."""

        self.shape_collision_group: wp.array[wp.int32] | None = None
        """Collision group of each shape, shape [shape_count], int. Array populated during finalization."""
        self.shape_collision_filter_pairs = _DeprecatedShapeCollisionFilterSet()
        """Set of canonical shape index pairs that should not collide.

        .. deprecated:: 1.4
            Mutating or reassigning this finalized-model set is deprecated. Configure collision
            filters on :class:`ModelBuilder` before calling :meth:`ModelBuilder.finalize` instead;
            post-finalize changes do not rebuild :attr:`shape_contact_pairs`.
        """
        self.shape_collision_radius: wp.array[wp.float32] | None = None
        """Collision radius [m] for bounding sphere broadphase, shape [shape_count], float. Not supported by :class:`~newton.solvers.SolverMuJoCo`."""
        self.shape_contact_pairs: wp.array[wp.vec2i] | None = None
        """Pairs of shape indices that may collide, shape [contact_pair_count, 2], int.

        Static-static pairs are omitted. Kinematic-kinematic and static-kinematic pairs
        are retained so consumers can opt into them during contact generation.
        """
        self.shape_contact_pair_count: int = 0
        """Number of shape contact pairs."""
        self.shape_world: wp.array[wp.int32] | None = None
        """World index for each shape, shape [shape_count], int. -1 for global."""
        self.shape_world_start: wp.array[wp.int32] | None = None
        """Start index of the first shape per world, shape [world_count + 2], int.

        The entries at indices ``0`` to ``world_count - 1`` store the start index of
        the shapes belonging to that world. The second-last element (accessible via
        index ``-2``) stores the start index of the global shapes (i.e. with world
        index ``-1``) added to the end of the model, and the last element stores the
        total shape count.

        The number of shapes in a given world ``w`` can be computed as::

            num_shapes_in_world = shape_world_start[w + 1] - shape_world_start[w]

        The total number of global shapes can be computed as::

            num_global_shapes = shape_world_start[-1] - shape_world_start[-2] + shape_world_start[0]
        """

        # Gaussians
        self.gaussians_count = 0
        """Number of gaussians."""
        self.gaussians_data = None
        """Data for Gaussian Splats, shape [gaussians_count], Gaussian.Data."""

        # Shape and particle BVH structures and related fields
        self.bvh_shapes: wp.Bvh | None = None
        """BVH over visible shapes, indexed by ``bvh_shape_enabled``. Built by :meth:`ModelBuilder.finalize`."""
        self.bvh_shapes_group_roots: wp.array[wp.int32] | None = None
        """Per-world BVH group roots for shapes, shape ``[world_count + 1]`` (last slot is global)."""
        self.bvh_shape_enabled: wp.array[wp.uint32] | None = None
        """Shape indices included in the shape BVH, shape ``[bvh_shape_count_enabled]``."""
        self.bvh_shape_count_enabled: int = 0
        """Number of shapes included in the shape BVH."""
        self.bvh_shape_bounds: wp.array2d[wp.vec3f] | None = None
        """Local-space AABB per shape (min/max) for mesh and gaussian shapes, shape ``[shape_count, 2]`` [m]."""
        self.bvh_shape_world_transforms: wp.array[wp.transformf] | None = None
        """World-space shape transforms computed during shape BVH build/refit, shape ``[shape_count]`` [m, unitless quaternion]."""

        self.bvh_particles: wp.Bvh | None = None
        """BVH over particles. Built by :meth:`ModelBuilder.finalize` when particles are present."""
        self.bvh_particles_group_roots: wp.array[wp.int32] | None = None
        """Per-world BVH group roots for particles, shape ``[world_count + 1]`` (last slot is global)."""

        # Heightfield collision data (compact table + per-shape index indirection)
        self.heightfield_count: int = 0
        """Number of ``GeoType.HFIELD`` shapes in the model."""
        self.shape_heightfield_index: wp.array[wp.int32] | None = None
        """Per-shape heightfield index, shape [shape_count]. -1 means shape has no heightfield."""
        self.heightfield_data: wp.array[HeightfieldData] | None = None
        """Compact array of HeightfieldData structs, one per actual heightfield shape."""
        self.heightfield_elevations: wp.array[wp.float32] | None = None
        """Concatenated 1D elevation array for all heightfields. Kernels index via HeightfieldData.data_offset."""
        self.heightfield_meshes: list[wp.Mesh] = []
        """wp.Mesh objects built from heightfield shapes, kept alive for the model's lifetime."""

        # Mesh edge data (packed array + per-shape slice)
        self.mesh_edge_indices: wp.array[wp.vec2i] | None = None
        """Packed unique edge vertex pairs for all mesh shapes, shape [total_edge_count]."""
        self.shape_edge_range: wp.array[wp.vec2i] | None = None
        """Per-shape (start, count) into mesh_edge_indices, shape [shape_count]. (-1,0) if no edges."""
        self._shape_mesh_properties: wp.array[wp.int32] | None = None
        """Per-shape mesh property bitfield used by collision kernels, shape [shape_count]."""

        # SDF storage (compact table + per-shape index indirection).
        # All SDF arrays are private; the public attribute names are exposed
        # via deprecated property aliases further down for back-compat.
        #
        # .. experimental::
        #     The SDF storage on ``Model`` is part of the experimental SDF API
        #     (see :class:`~newton.SDF`) and may change without notice.
        self._shape_sdf_index: wp.array[wp.int32] | None = None
        """Per-shape SDF index, shape [shape_count]. -1 means shape has no SDF."""

        # Texture SDF storage
        self._texture_sdf_data = None
        """Compact array of TextureSDFData structs, shape [num_sdfs]."""
        self._texture_sdf_coarse_textures: list = []
        """Coarse 3D textures matching _texture_sdf_data by index. Kept for reference counting."""
        self._texture_sdf_subgrid_textures: list = []
        """Subgrid 3D textures matching _texture_sdf_data by index. Kept for reference counting."""
        self._texture_sdf_subgrid_start_slots: list = []
        """Subgrid start slot arrays matching _texture_sdf_data by index. Kept for reference counting."""

        # Caches for the deprecated lazy ``sdf_block_coords`` / ``sdf_index2blocks``
        # properties. Populated on first access; cleared when SDF storage changes.
        self._sdf_block_coords_cache: wp.array | None = None
        self._sdf_index2blocks_cache: wp.array | None = None

        # Local AABB and voxel grid for contact reduction
        # Note: These are stored in Model (not Contacts) because they are static geometry properties
        # computed once during finalization, not per-frame contact data.
        self.shape_collision_aabb_lower: wp.array[wp.vec3] | None = None
        """Scaled local-space AABB lower bound [m] for each shape, shape [shape_count, 3], float.
        Includes shape scale but excludes margin and gap (those are applied at runtime).
        Used for broadphase AABB computation and voxel-based contact reduction."""
        self.shape_collision_aabb_upper: wp.array[wp.vec3] | None = None
        """Scaled local-space AABB upper bound [m] for each shape, shape [shape_count, 3], float.
        Includes shape scale but excludes margin and gap (those are applied at runtime).
        Used for broadphase AABB computation and voxel-based contact reduction."""
        self._shape_voxel_resolution: wp.array[wp.vec3i] | None = None
        """Voxel grid resolution (nx, ny, nz) for each shape, shape [shape_count, 3], int. Used for voxel-based contact reduction."""

        self.spring_indices: wp.array[wp.int32] | None = None
        """Particle spring indices, shape [spring_count*2], int."""
        self.spring_rest_length: wp.array[wp.float32] | None = None
        """Particle spring rest length [m], shape [spring_count], float."""
        self.spring_stiffness: wp.array[wp.float32] | None = None
        """Particle spring stiffness [N/m], shape [spring_count], float."""
        self.spring_damping: wp.array[wp.float32] | None = None
        """Particle spring damping [N·s/m], shape [spring_count], float."""
        self.spring_control: wp.array[wp.float32] | None = None
        """Particle spring activation [dimensionless], shape [spring_count], float."""
        self.spring_constraint_lambdas: wp.array[wp.float32] | None = None
        """Lagrange multipliers for spring constraints (internal use)."""

        self.tri_indices: wp.array[wp.int32] | None = None
        """Triangle element indices, shape [tri_count*3], int."""
        self.tri_poses: wp.array[wp.mat22] | None = None
        """Triangle element rest pose, shape [tri_count, 2, 2], float."""
        self.tri_activations: wp.array[wp.float32] | None = None
        """Triangle element activations, shape [tri_count], float."""
        self.tri_materials: wp.array2d[wp.float32] | None = None
        """Triangle element materials, shape [tri_count, 5], float.
        Components: [0] k_mu [Pa], [1] k_lambda [Pa], [2] k_damp [Pa·s], [3] k_drag [Pa·s], [4] k_lift [Pa].
        Stored per-element; kernels multiply by rest area internally."""
        self.tri_areas: wp.array[wp.float32] | None = None
        """Triangle element rest areas [m²], shape [tri_count], float."""

        self.edge_indices: wp.array[wp.int32] | None = None
        """Bending edge indices, shape [edge_count*4], int, each row is [o0, o1, v1, v2], where v1, v2 are on the edge."""
        self.edge_rest_angle: wp.array[wp.float32] | None = None
        """Bending edge rest angle [rad], shape [edge_count], float."""
        self.edge_rest_length: wp.array[wp.float32] | None = None
        """Bending edge rest length [m], shape [edge_count], float."""
        self.edge_bending_properties: wp.array2d[wp.float32] | None = None
        """Bending edge stiffness and damping, shape [edge_count, 2], float.
        Components: [0] stiffness [N·m/rad], [1] damping [N·s]."""
        self.edge_constraint_lambdas: wp.array[wp.float32] | None = None
        """Lagrange multipliers for edge constraints (internal use)."""
        self.soft_mesh_adjacency: MeshAdjacency | None = None
        """Soft mesh topology and solver adjacency, or ``None`` before finalization."""
        self.soft_mesh_adjacency_device: MeshAdjacencyData | None = None
        """Device-uploaded :attr:`soft_mesh_adjacency`, built once at finalization and shared by all
        consumers (VBD solver, collision pipeline). ``None`` before finalization."""

        self.tet_indices: wp.array[wp.int32] | None = None
        """Tetrahedral element indices, shape [tet_count*4], int."""
        self.tet_poses: wp.array[wp.mat33] | None = None
        """Tetrahedral rest poses, shape [tet_count, 3, 3], float."""
        self.tet_activations: wp.array[wp.float32] | None = None
        """Tetrahedral volumetric activations, shape [tet_count], float."""
        self.tet_materials: wp.array2d[wp.float32] | None = None
        """Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3].
        Components: [0] k_mu [Pa], [1] k_lambda [Pa], [2] k_damp [Pa·s].
        Stored per-element; kernels multiply by rest volume internally."""

        self.muscle_start: wp.array[wp.int32] | None = None
        """Start index of the first muscle point per muscle, shape [muscle_count], int."""
        self.muscle_params: wp.array2d[wp.float32] | None = None
        """Muscle parameters, shape [muscle_count, 5], float.
        Components: [0] f0 [N] (force scaling), [1] lm [m] (muscle fiber length), [2] lt [m] (tendon slack length),
        [3] lmax [m] (max efficient length), [4] pen [dimensionless] (penalty factor)."""
        self.muscle_bodies: wp.array[wp.int32] | None = None
        """Body indices of the muscle waypoints, int."""
        self.muscle_points: wp.array[wp.vec3] | None = None
        """Local body offset of the muscle waypoints, float."""
        self.muscle_activations: wp.array[wp.float32] | None = None
        """Muscle activations [dimensionless, 0 to 1], shape [muscle_count], float."""

        self.body_q: wp.array[wp.transform] | None = None
        """Rigid body poses [m, unitless quaternion] for state initialization, shape [body_count, 7], float."""
        self.body_qd: wp.array[wp.spatial_vector] | None = None
        """Rigid body velocities [m/s, rad/s] for state initialization, shape [body_count, 6], float.
        The linear component is the body COM velocity in world frame."""
        self.body_com: wp.array[wp.vec3] | None = None
        """Rigid body center of mass [m] (in local frame), shape [body_count, 3], float."""
        self.body_inertia: wp.array[wp.mat33] | None = None
        """Rigid body inertia tensor [kg·m²] (relative to COM), shape [body_count, 3, 3], float."""
        self.body_inv_inertia: wp.array[wp.mat33] | None = None
        """Rigid body inverse inertia tensor [1/(kg·m²)] (relative to COM), shape [body_count, 3, 3], float."""
        self.body_mass: wp.array[wp.float32] | None = None
        """Rigid body mass [kg], shape [body_count], float."""
        self.body_inv_mass: wp.array[wp.float32] | None = None
        """Rigid body inverse mass [1/kg], shape [body_count], float."""
        self.body_flags: wp.array[wp.int32] | None = None
        """Rigid body flags (:class:`~newton.BodyFlags`), shape [body_count], int."""
        self.body_label: list[str] = []
        """Rigid body labels, shape [body_count], str."""
        self.body_world: wp.array[wp.int32] | None = None
        """World index for each body, shape [body_count], int. Global entities have index -1."""
        self.body_world_start: wp.array[wp.int32] | None = None
        """Start index of the first body per world, shape [world_count + 2], int.

        The entries at indices ``0`` to ``world_count - 1`` store the start index of
        the bodies belonging to that world. The second-last element (accessible via
        index ``-2``) stores the start index of the global bodies (i.e. with world
        index ``-1``) added to the end of the model, and the last element stores the
        total body count.

        The number of bodies in a given world ``w`` can be computed as::

            num_bodies_in_world = body_world_start[w + 1] - body_world_start[w]

        The total number of global bodies can be computed as::

            num_global_bodies = body_world_start[-1] - body_world_start[-2] + body_world_start[0]
        """

        self.joint_q: wp.array[wp.float32] | None = None
        """Generalized joint positions [m or rad, depending on joint type] for state initialization, shape [joint_coord_count], float."""
        self.joint_qd: wp.array[wp.float32] | None = None
        """Generalized joint velocities [m/s or rad/s, depending on joint type] for state initialization, shape [joint_dof_count], float.
        For FREE and DISTANCE joints, the linear entries are child-COM velocity in the joint parent frame and the angular entries are angular velocity in that same frame."""
        self.joint_f: wp.array[wp.float32] | None = None
        """Default generalized joint forces [N or N·m, depending on joint type] used to initialize :attr:`newton.Control.joint_f`, shape [joint_dof_count], float.
        For FREE and DISTANCE joints, the linear entries are world-frame force at the child COM and the angular entries are world-frame torque about the child COM."""
        self.joint_target_q: wp.array[wp.float32] | None = None
        """Generalized joint position targets [m or rad, depending on joint type] used to initialize :attr:`newton.Control.joint_target_q`, shape ``[joint_coord_count]`` or ``[joint_dof_count]``, float.

        Shape matches :attr:`joint_q` (``joint_coord_count``) when
        :attr:`newton.use_coord_layout_targets` is ``True``; otherwise the array
        is shaped ``(joint_dof_count,)`` for backward compatibility with the
        deprecated :attr:`joint_target_pos` alias. Index via
        :attr:`joint_target_q_start`, which aliases :attr:`joint_q_start` or
        :attr:`joint_qd_start` to match the active layout.
        """
        self.joint_target_qd: wp.array[wp.float32] | None = None
        """Generalized joint velocity targets [m/s or rad/s, depending on joint type] used to initialize :attr:`newton.Control.joint_target_qd`, shape [joint_dof_count], float.

        Matches the layout of :attr:`joint_qd`. Replaces the deprecated
        :attr:`joint_target_vel`.
        """
        self.joint_act: wp.array[wp.float32] | None = None
        """Per-DOF feedforward actuation input for control initialization, shape [joint_dof_count], float."""
        self.joint_type: wp.array[wp.int32] | None = None
        """Joint type, shape [joint_count], int."""
        self._has_cable_joints: bool = False
        self.joint_articulation: wp.array[wp.int32] | None = None
        """Joint articulation index (-1 if not in any articulation), shape [joint_count], int."""
        self.joint_parent: wp.array[wp.int32] | None = None
        """Joint parent body indices, shape [joint_count], int."""
        self.joint_child: wp.array[wp.int32] | None = None
        """Joint child body indices, shape [joint_count], int."""
        self.joint_ancestor: wp.array[wp.int32] | None = None
        """Maps from joint index to the index of the joint that has the current joint parent body as child (-1 if no such joint ancestor exists), shape [joint_count], int."""
        self.joint_X_p: wp.array[wp.transform] | None = None
        """Joint transform in parent frame [m, unitless quaternion], shape [joint_count, 7], float."""
        self.joint_X_c: wp.array[wp.transform] | None = None
        """Joint mass frame in child frame [m, unitless quaternion], shape [joint_count, 7], float."""
        self.joint_axis: wp.array[wp.vec3] | None = None
        """Joint axis in child frame, shape [joint_dof_count, 3], float."""
        self.joint_armature: wp.array[wp.float32] | None = None
        """Armature [kg·m² (rotational) or kg (translational)] for each joint axis (used by :class:`~newton.solvers.SolverMuJoCo` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_target_mode: wp.array[wp.int32] | None = None
        """Joint target mode per DOF, see :class:`newton.JointTargetMode`. Shape [joint_dof_count], dtype int32."""
        self.joint_target_ke: wp.array[wp.float32] | None = None
        """Joint stiffness [N/m or N·m/rad, depending on joint type], shape [joint_dof_count], float."""
        self.joint_target_kd: wp.array[wp.float32] | None = None
        """Joint damping [N·s/m or N·m·s/rad, depending on joint type], shape [joint_dof_count], float."""
        self.joint_damping: wp.array[wp.float32] | None = None
        """Passive velocity damping [N·s/m or N·m·s/rad, depending on joint type] always active on the joint, shape [joint_dof_count], float."""
        self.joint_effort_limit: wp.array[wp.float32] | None = None
        """Joint effort (force/torque) limits [N or N·m, depending on joint type], shape [joint_dof_count], float."""
        self.joint_velocity_limit: wp.array[wp.float32] | None = None
        """Joint velocity limits [m/s or rad/s, depending on joint type], shape [joint_dof_count], float."""
        self.joint_friction: wp.array[wp.float32] | None = None
        """Joint friction force/torque [N or N·m, depending on joint type], shape [joint_dof_count], float."""
        self.joint_dof_dim: wp.array2d[wp.int32] | None = None
        """Number of linear and angular dofs per joint, shape [joint_count, 2], int."""
        self.joint_enabled: wp.array[wp.bool] | None = None
        """Controls which joint is simulated (bodies become disconnected if False, supported by :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverVBD`, and :class:`~newton.solvers.SolverSemiImplicit`), shape [joint_count], bool."""
        self.joint_limit_lower: wp.array[wp.float32] | None = None
        """Joint lower position limits [m or rad, depending on joint type], shape [joint_dof_count], float. Values must be finite; use ``-newton.MAXVAL`` to indicate no lower limit."""
        self.joint_limit_upper: wp.array[wp.float32] | None = None
        """Joint upper position limits [m or rad, depending on joint type], shape [joint_dof_count], float. Values must be finite; use ``newton.MAXVAL`` to indicate no upper limit."""
        self.joint_limit_ke: wp.array[wp.float32] | None = None
        """Joint position limit stiffness [N/m or N·m/rad, depending on joint type] (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_limit_kd: wp.array[wp.float32] | None = None
        """Joint position limit damping [N·s/m or N·m·s/rad, depending on joint type] (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_twist_lower: wp.array[wp.float32] | None = None
        """Joint lower twist limit [rad], shape [joint_count], float."""
        self.joint_twist_upper: wp.array[wp.float32] | None = None
        """Joint upper twist limit [rad], shape [joint_count], float."""
        self.joint_q_start: wp.array[wp.int32] | None = None
        """Start index of the first position coordinate per joint (last value is a sentinel for dimension queries), shape [joint_count + 1], int."""
        self.joint_qd_start: wp.array[wp.int32] | None = None
        """Start index of the first velocity coordinate per joint (last value is a sentinel for dimension queries), shape [joint_count + 1], int."""
        self.joint_label: list[str] = []
        """Joint labels, shape [joint_count], str."""
        self.joint_world: wp.array[wp.int32] | None = None
        """World index for each joint, shape [joint_count], int. -1 for global."""
        self.joint_world_start: wp.array[wp.int32] | None = None
        """Start index of the first joint per world, shape [world_count + 2], int.

        The entries at indices ``0`` to ``world_count - 1`` store the start index of
        the joints belonging to that world. The second-last element (accessible via
        index ``-2``) stores the start index of the global joints (i.e. with world
        index ``-1``) added to the end of the model, and the last element stores the
        total joint count.

        The number of joints in a given world ``w`` can be computed as::

            num_joints_in_world = joint_world_start[w + 1] - joint_world_start[w]

        The total number of global joints can be computed as::

            num_global_joints = joint_world_start[-1] - joint_world_start[-2] + joint_world_start[0]
        """
        self.joint_dof_world_start: wp.array[wp.int32] | None = None
        """Start index of the first joint degree of freedom per world, shape [world_count + 2], int.

        The entries at indices ``0`` to ``world_count - 1`` store the start index of
        the joint DOFs belonging to that world. The second-last element (accessible
        via index ``-2``) stores the start index of the global joint DOFs (i.e. with
        world index ``-1``) added to the end of the model, and the last element
        stores the total joint DOF count.

        The number of joint DOFs in a given world ``w`` can be computed as::

            num_joint_dofs_in_world = joint_dof_world_start[w + 1] - joint_dof_world_start[w]

        The total number of global joint DOFs can be computed as::

            num_global_joint_dofs = joint_dof_world_start[-1] - joint_dof_world_start[-2] + joint_dof_world_start[0]
        """
        self.joint_coord_world_start: wp.array[wp.int32] | None = None
        """Start index of the first joint coordinate per world, shape [world_count + 2], int.

        The entries at indices ``0`` to ``world_count - 1`` store the start index of
        the joint coordinates belonging to that world. The second-last element
        (accessible via index ``-2``) stores the start index of the global joint
        coordinates (i.e. with world index ``-1``) added to the end of the model,
        and the last element stores the total joint coordinate count.

        The number of joint coordinates in a given world ``w`` can be computed as::

            num_joint_coords_in_world = joint_coord_world_start[w + 1] - joint_coord_world_start[w]

        The total number of global joint coordinates can be computed as::

            num_global_joint_coords = joint_coord_world_start[-1] - joint_coord_world_start[-2] + joint_coord_world_start[0]
        """
        self.joint_constraint_world_start: wp.array[wp.int32] | None = None
        """Start index of the first joint constraint per world, shape [world_count + 2], int.

        The entries at indices ``0`` to ``world_count - 1`` store the start index of
        the joint constraints belonging to that world. The second-last element
        (accessible via index ``-2``) stores the start index of the global joint
        constraints (i.e. with world index ``-1``) added to the end of the model,
        and the last element stores the total joint constraint count.

        The number of joint constraints in a given world ``w`` can be computed as::

            num_joint_constraints_in_world = joint_constraint_world_start[w + 1] - joint_constraint_world_start[w]

        The total number of global joint constraints can be computed as::

            num_global_joint_constraints = joint_constraint_world_start[-1] - joint_constraint_world_start[-2] + joint_constraint_world_start[0]
        """

        self.articulation_start: wp.array[wp.int32] | None = None
        """Articulation start index plus sentinel, shape [articulation_count + 1], int.

        The sentinel still bounds each articulation's full joint range, including
        converted loop-closing joints. Use :attr:`articulation_end` for the
        exclusive end of regular tree joints.
        """
        self.articulation_end: wp.array[wp.int32] | None = None
        """Exclusive end index of regular tree joints per articulation, shape [articulation_count], int."""
        self.articulation_label: list[str] = []
        """Articulation labels, shape [articulation_count], str."""
        self.articulation_world: wp.array[wp.int32] | None = None
        """World index for each articulation, shape [articulation_count], int. -1 for global."""
        self.articulation_world_start: wp.array[wp.int32] | None = None
        """Start index of the first articulation per world, shape [world_count + 2], int.

        The entries at indices ``0`` to ``world_count - 1`` store the start index of
        the articulations belonging to that world. The second-last element
        (accessible via index ``-2``) stores the start index of the global
        articulations (i.e. with world index ``-1``) added to the end of the model,
        and the last element stores the total articulation count.

        The number of articulations in a given world ``w`` can be computed as::

            num_articulations_in_world = articulation_world_start[w + 1] - articulation_world_start[w]

        The total number of global articulations can be computed as::

            num_global_articulations = articulation_world_start[-1] - articulation_world_start[-2] + articulation_world_start[0]
        """
        self.max_joints_per_articulation: int = 0
        """Maximum number of joints in any articulation (used for IK kernel dimensioning)."""
        self.max_dofs_per_articulation: int = 0
        """Maximum number of degrees of freedom in any articulation (used for Jacobian/mass matrix computation)."""

        self.soft_contact_ke: float = 1.0e3
        """Stiffness of soft contacts [N/m] (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_kd: float = 10.0
        """Damping of soft contacts [N·s/m] (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_kf: float = 1.0e3
        """Soft contact friction gain [N·s/m] (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_mu: float = 0.5
        """Friction coefficient of soft contacts [dimensionless]."""
        self.soft_contact_restitution: float = 0.0
        """Restitution coefficient of soft contacts [dimensionless] (used by :class:`SolverXPBD`)."""

        self.rigid_contact_max: int = 0
        """Number of potential contact points between rigid bodies."""

        self.up_axis: int = 2
        """Up axis: 0 for x, 1 for y, 2 for z."""
        self.gravity: wp.array[wp.vec3] | None = None
        """Per-world gravity vectors [m/s²], shape [world_count, 3], dtype :class:`vec3`."""

        self.constraint_mimic_joint0: wp.array[wp.int32] | None = None
        """Follower joint index (``joint0 = coef0 + coef1 * joint1``), shape [constraint_mimic_count], int."""
        self.constraint_mimic_joint1: wp.array[wp.int32] | None = None
        """Leader joint index (``joint0 = coef0 + coef1 * joint1``), shape [constraint_mimic_count], int."""
        self.constraint_mimic_coef0: wp.array[wp.float32] | None = None
        """Offset coefficient (coef0) for the mimic constraint (``joint0 = coef0 + coef1 * joint1``), shape [constraint_mimic_count], float."""
        self.constraint_mimic_coef1: wp.array[wp.float32] | None = None
        """Scale coefficient (coef1) for the mimic constraint (``joint0 = coef0 + coef1 * joint1``), shape [constraint_mimic_count], float."""
        self.constraint_mimic_enabled: wp.array[wp.bool] | None = None
        """Whether constraint is active, shape [constraint_mimic_count], bool."""
        self.constraint_mimic_label: list[str] = []
        """Constraint name/label, shape [constraint_mimic_count], str."""
        self.constraint_mimic_world: wp.array[wp.int32] | None = None
        """World index for each constraint, shape [constraint_mimic_count], int."""

        self.particle_count: int = 0
        """Total number of particles in the system."""
        self.body_count: int = 0
        """Total number of bodies in the system."""
        self.shape_count: int = 0
        """Total number of shapes in the system."""
        self.joint_count: int = 0
        """Total number of joints in the system."""
        self.tri_count: int = 0
        """Total number of triangles in the system."""
        self.tet_count: int = 0
        """Total number of tetrahedra in the system."""
        self.edge_count: int = 0
        """Total number of edges in the system."""
        self.spring_count: int = 0
        """Total number of springs in the system."""
        self.muscle_count: int = 0
        """Total number of muscles in the system."""
        self.articulation_count: int = 0
        """Total number of articulations in the system."""
        self.joint_dof_count: int = 0
        """Total number of velocity degrees of freedom of all joints. Equals the number of joint axes."""
        self.joint_coord_count: int = 0
        """Total number of position degrees of freedom of all joints."""
        self.joint_constraint_count: int = 0
        """Total number of joint constraints of all joints."""
        self.constraint_mimic_count: int = 0
        """Total number of mimic constraints in the system."""

        # indices of particles sharing the same color
        self.particle_color_groups: list[wp.array[wp.int32]] = []
        """Coloring of all particles for Gauss-Seidel iteration (see :class:`~newton.solvers.SolverVBD`). Each array contains indices of particles sharing the same color."""
        self.particle_colors: wp.array[wp.int32] | None = None
        """Color assignment for every particle."""

        self.body_color_groups: list[wp.array[wp.int32]] = []
        """Coloring of all rigid bodies for Gauss-Seidel iteration (see :class:`~newton.solvers.SolverVBD`). Each array contains indices of bodies sharing the same color."""
        self.body_colors: wp.array[wp.int32] | None = None
        """Color assignment for every rigid body."""

        self.device: wp.Device = wp.get_device(device)
        """Device on which the Model was allocated."""

        import newton  # noqa: PLC0415

        self.use_coord_layout_targets: bool = newton.use_coord_layout_targets
        """Snapshot of :data:`newton.use_coord_layout_targets` taken at
        :meth:`ModelBuilder.finalize`. All layout decisions for this Model
        consult this — toggling the global later doesn't change behavior."""

        self.custom_frequency_counts: dict[str, int] = {}
        """Counts for custom frequencies (e.g., ``{"mujoco:pair": 5}``). Set during finalize()."""

        self._requested_state_attributes: set[str] = set()
        self._collision_pipeline: CollisionPipeline | None = None
        # cached collision pipeline
        self._requested_contact_attributes: set[str] = set()

        target_q_freq = (
            Model.AttributeFrequency.JOINT_COORD
            if self.use_coord_layout_targets
            else Model.AttributeFrequency.JOINT_DOF
        )
        self.attribute_specs: dict[str, Model.AttributeSpec] = dict(Model._CORE_ATTRIBUTE_SPECS)
        """Semantic metadata keyed by built-in or custom attribute name.

        Attribute values remain normal Python attributes on the model. This
        registry describes their indexing, ownership, and reference semantics.
        """

        self.attribute_specs["joint_target_q"] = Model.AttributeSpec(target_q_freq)
        if not self.use_coord_layout_targets:
            self.attribute_specs["joint_target_pos"] = Model.AttributeSpec(
                target_q_freq,
                deprecated=True,
                alias_of="joint_target_q",
            )
            self.attribute_specs["joint_target_vel"] = Model.AttributeSpec(
                Model.AttributeFrequency.JOINT_DOF,
                deprecated=True,
                alias_of="joint_target_qd",
            )

        # Extended state attributes live on State and are allocated only when
        # explicitly requested via request_state_attributes().
        for full_name, template in State.EXTENDED_ATTRIBUTE_TEMPLATES.items():
            self.attribute_specs[full_name] = Model.AttributeSpec(getattr(Model.AttributeFrequency, template.frequency))

        self.attribute_frequency: dict[str, Model.AttributeFrequency | str] = {
            name: spec.frequency for name, spec in self.attribute_specs.items()
        }
        """Compatibility map from attribute names to their indexing frequencies."""

        self.attribute_assignment: dict[str, Model.AttributeAssignment] = {
            name: spec.assignment for name, spec in self.attribute_specs.items() if spec.assignment is not None
        }
        """Compatibility map from custom attributes to their assignment categories."""

        self.actuators: list[Actuator] = []
        """List of actuator instances for this model."""

    def _set_shape_collision_filter_packed(self, packed: np.ndarray) -> None:
        """Install the canonical filter store: sorted unique packed pair codes."""
        self.__dict__["shape_collision_filter_pairs"] = _DeprecatedShapeCollisionFilterSet(packed=packed)

    def _shape_collision_filter_store(self) -> _DeprecatedShapeCollisionFilterSet | None:
        """Return the stored filter view without triggering materialization.

        The store shares the instance-dict slot with the public
        ``shape_collision_filter_pairs`` descriptor, whose ``__get__``
        materializes the set; array-backed queries read the slot directly so
        they stay materialization-free.
        """
        return self.__dict__.get("shape_collision_filter_pairs")

    def shape_collision_filter_contains(self, shape_a: SupportsIndex, shape_b: SupportsIndex) -> bool:
        """Return whether a canonicalized shape pair is collision-filtered.

        This queries the canonical filter-pair array when available, avoiding
        materialization of :attr:`shape_collision_filter_pairs`.

        Args:
            shape_a: First shape index.
            shape_b: Second shape index.

        Returns:
            Whether the pair is present in the collision filter.

        Raises:
            TypeError: If either shape index is not an integer.
        """
        filters = self._shape_collision_filter_store()
        if filters is None:
            return False
        return filters.contains_pair(shape_a, shape_b)

    def shape_collision_filter_pairs_array(self) -> np.ndarray:
        """Return the collision-filter pairs as an array.

        Array counterpart to :attr:`shape_collision_filter_pairs` that returns
        the canonical filter-pair array without materializing the public set.
        Consumers that need every excluded pair — such as the ``"nxn"`` and
        ``"sap"`` broad-phase exclusion arrays — should prefer this form.

        Returns:
            Canonical shape index pairs sorted lexicographically, shape
            [pair_count, 2].
        """
        filters = self._shape_collision_filter_store()
        if filters is None:
            return np.empty((0, 2), dtype=np.int32)
        return filters.pairs_array()

    def shape_collision_filter_mask(self, pairs: np.ndarray) -> np.ndarray:
        """Return a boolean mask of which shape pairs are collision-filtered.

        Bulk counterpart to :meth:`shape_collision_filter_contains`: one
        vectorized query against the canonical filter-pair array instead of a
        Python-level membership test per pair.

        Args:
            pairs: Shape index pairs in any order, shape [pair_count, 2].

        Returns:
            Boolean mask of filtered pairs, shape [pair_count].

        Raises:
            TypeError: If ``pairs`` does not have an integer dtype.
            OverflowError: If an unsigned shape index exceeds the signed 64-bit range.
        """
        pairs = np.asarray(pairs)
        if pairs.dtype.kind not in "iu":
            raise TypeError(f"pairs must have an integer dtype, got {pairs.dtype}")
        if pairs.dtype.kind == "u":
            if pairs.size > 0 and np.any(pairs > np.iinfo(np.int64).max):
                raise OverflowError("unsigned shape indices must fit in a signed 64-bit integer")
            pairs = pairs.astype(np.int64)
        pairs = pairs.reshape((-1, 2))
        filters = self._shape_collision_filter_store()
        if filters is None:
            return np.zeros(pairs.shape[0], dtype=bool)
        return filters.mask_pairs(pairs)

    def _attribute_spec(self, name: str) -> Model.AttributeSpec | None:
        """Return current metadata, including legacy mapping overrides."""
        spec = self.attribute_specs.get(name)
        frequency = self.attribute_frequency.get(name, None if spec is None else spec.frequency)
        if frequency is None:
            return None
        assignment = self.attribute_assignment.get(name, None if spec is None else spec.assignment)
        if spec is None:
            return Model.AttributeSpec(frequency=frequency, assignment=assignment)
        if frequency != spec.frequency or assignment != spec.assignment:
            return replace(spec, frequency=frequency, assignment=assignment)
        return spec

    def _iter_attribute_specs(self, *, include_deprecated: bool = False) -> Iterator[tuple[str, Model.AttributeSpec]]:
        """Yield unified metadata, including late legacy registrations.

        Args:
            include_deprecated: Whether to include deprecated compatibility aliases.
        """
        names = dict.fromkeys((*self.attribute_specs, *self.attribute_frequency))
        for name in names:
            spec = self._attribute_spec(name)
            if spec is not None and (include_deprecated or not spec.deprecated):
                yield name, spec

    def _set_attribute_spec(self, name: str, spec: Model.AttributeSpec) -> None:
        """Register unified metadata and update the legacy compatibility maps."""
        self.attribute_specs[name] = spec
        self.attribute_frequency[name] = spec.frequency
        if spec.assignment is None:
            self.attribute_assignment.pop(name, None)
        else:
            self.attribute_assignment[name] = spec.assignment

    def _resolve_attribute_frequency(self, name: str) -> Model.AttributeFrequency | str | None:
        """Return explicitly registered frequency metadata."""
        spec = self._attribute_spec(name)
        return None if spec is None else spec.frequency

    def _attribute_reference_frequency(self, name: str) -> Model.AttributeFrequency | str | None:
        """Return the entity domain indexed by an attribute's values."""
        spec = self._attribute_spec(name)
        return None if spec is None else spec.references

    def _attribute_row_width(self, name: str) -> int:
        """Return the number of flattened values stored per frequency row."""
        spec = self._attribute_spec(name)
        return 1 if spec is None else spec.row_width

    def _attribute_requires_empty_sentinel(self, name: str) -> bool:
        """Return whether an empty attribute retains one sentinel value."""
        spec = self._attribute_spec(name)
        return False if spec is None else spec.requires_empty_sentinel

    def _normalize_attribute_reference(self, references: str | None) -> Model.AttributeFrequency | str | None:
        """Return the frequency domain addressed by a builder reference declaration."""
        if references is None:
            return None
        built_in = {
            "body": Model.AttributeFrequency.BODY,
            "shape": Model.AttributeFrequency.SHAPE,
            "joint": Model.AttributeFrequency.JOINT,
            "joint_dof": Model.AttributeFrequency.JOINT_DOF,
            "joint_coord": Model.AttributeFrequency.JOINT_COORD,
            "joint_constraint": Model.AttributeFrequency.JOINT_CONSTRAINT,
            "articulation": Model.AttributeFrequency.ARTICULATION,
            "equality_constraint": "mujoco:equality_constraint",
            "constraint_mimic": Model.AttributeFrequency.CONSTRAINT_MIMIC,
            "particle": Model.AttributeFrequency.PARTICLE,
            "edge": Model.AttributeFrequency.EDGE,
            "triangle": Model.AttributeFrequency.TRIANGLE,
            "tetrahedron": Model.AttributeFrequency.TETRAHEDRON,
            "spring": Model.AttributeFrequency.SPRING,
            "world": Model.AttributeFrequency.WORLD,
        }
        frequency = built_in.get(references)
        if frequency is not None:
            return frequency
        if references in self.custom_frequency_counts:
            return references
        raise ValueError(f"Unknown custom attribute reference frequency {references!r}")

    # ----- Deprecated SDF aliases -------------------------------------------
    # The underlying SDF members on ``Model`` are now underscore-prefixed.
    # The properties below preserve the historical attribute names for one
    # release cycle and emit ``DeprecationWarning`` on access.

    @property
    def shape_sdf_index(self) -> wp.array[wp.int32] | None:
        """Deprecated alias for :attr:`_shape_sdf_index`.

        .. deprecated:: 1.3
            Use the underscored private member or the appropriate accessor.
            This alias will be removed in a future release.
        """
        warnings.warn(
            "Model.shape_sdf_index is deprecated; use Model._shape_sdf_index. "
            "The public alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._shape_sdf_index

    @shape_sdf_index.setter
    def shape_sdf_index(self, value):
        warnings.warn(
            "Model.shape_sdf_index is deprecated; assign to Model._shape_sdf_index. "
            "The public alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._shape_sdf_index = value

    @property
    def texture_sdf_data(self):
        """Deprecated alias for :attr:`_texture_sdf_data`.

        .. deprecated:: 1.3
            Use the underscored private member. The alias will be removed in
            a future release.
        """
        warnings.warn(
            "Model.texture_sdf_data is deprecated; use Model._texture_sdf_data. "
            "The public alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._texture_sdf_data

    @texture_sdf_data.setter
    def texture_sdf_data(self, value):
        warnings.warn(
            "Model.texture_sdf_data is deprecated; assign to Model._texture_sdf_data. "
            "The public alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._texture_sdf_data = value
        self._sdf_block_coords_cache = None
        self._sdf_index2blocks_cache = None

    @property
    def texture_sdf_coarse_textures(self) -> list:
        """Deprecated alias for :attr:`_texture_sdf_coarse_textures`.

        .. deprecated:: 1.3
            Use the underscored private member. The alias will be removed in
            a future release.
        """
        warnings.warn(
            "Model.texture_sdf_coarse_textures is deprecated; use "
            "Model._texture_sdf_coarse_textures. The public alias will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._texture_sdf_coarse_textures

    @texture_sdf_coarse_textures.setter
    def texture_sdf_coarse_textures(self, value):
        warnings.warn(
            "Model.texture_sdf_coarse_textures is deprecated; assign to "
            "Model._texture_sdf_coarse_textures. The public alias will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._texture_sdf_coarse_textures = value
        self._sdf_block_coords_cache = None
        self._sdf_index2blocks_cache = None

    @property
    def texture_sdf_subgrid_textures(self) -> list:
        """Deprecated alias for :attr:`_texture_sdf_subgrid_textures`.

        .. deprecated:: 1.3
            Use the underscored private member. The alias will be removed in
            a future release.
        """
        warnings.warn(
            "Model.texture_sdf_subgrid_textures is deprecated; use "
            "Model._texture_sdf_subgrid_textures. The public alias will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._texture_sdf_subgrid_textures

    @texture_sdf_subgrid_textures.setter
    def texture_sdf_subgrid_textures(self, value):
        warnings.warn(
            "Model.texture_sdf_subgrid_textures is deprecated; assign to "
            "Model._texture_sdf_subgrid_textures. The public alias will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._texture_sdf_subgrid_textures = value

    @property
    def texture_sdf_subgrid_start_slots(self) -> list:
        """Deprecated alias for :attr:`_texture_sdf_subgrid_start_slots`.

        .. deprecated:: 1.3
            Use the underscored private member. The alias will be removed in
            a future release.
        """
        warnings.warn(
            "Model.texture_sdf_subgrid_start_slots is deprecated; use "
            "Model._texture_sdf_subgrid_start_slots. The public alias will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._texture_sdf_subgrid_start_slots

    @texture_sdf_subgrid_start_slots.setter
    def texture_sdf_subgrid_start_slots(self, value):
        warnings.warn(
            "Model.texture_sdf_subgrid_start_slots is deprecated; assign to "
            "Model._texture_sdf_subgrid_start_slots. The public alias will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._texture_sdf_subgrid_start_slots = value

    @property
    def sdf_block_coords(self):
        """Deprecated.  Lazily-computed flat ``wp.vec3us`` block coords.

        Per-SDF active-block coordinates were dropped when the hydroelastic
        broadphase started deriving them arithmetically from each SDF's
        coarse-texture dimensions. This property recomputes the legacy
        layout on first access (and caches it) so external callers that
        still read the attribute keep working.

        .. deprecated:: 1.3
            This attribute will be removed in a future release.
        """
        warnings.warn(
            "Model.sdf_block_coords is deprecated and will be removed in "
            "a future release. The hydroelastic broadphase now derives block "
            "coordinates arithmetically from each SDF's coarse-texture "
            "dimensions and no longer needs this attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._ensure_legacy_sdf_block_arrays()
        return self._sdf_block_coords_cache

    @property
    def sdf_index2blocks(self):
        """Deprecated.  Lazily-computed per-SDF ``[start, end)`` ranges.

        Per-SDF ``[start, end)`` indices into ``sdf_block_coords`` were
        dropped when the hydroelastic broadphase started deriving block
        ranges arithmetically from each SDF's coarse-texture dimensions.
        This property recomputes the legacy layout on first access (and
        caches it) so external callers that still read the attribute keep
        working.

        .. deprecated:: 1.3
            This attribute will be removed in a future release.
        """
        warnings.warn(
            "Model.sdf_index2blocks is deprecated and will be removed in "
            "a future release. The hydroelastic broadphase now derives block "
            "ranges arithmetically from each SDF's coarse-texture "
            "dimensions and no longer needs this attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._ensure_legacy_sdf_block_arrays()
        return self._sdf_index2blocks_cache

    def _ensure_legacy_sdf_block_arrays(self) -> None:
        """Populate the legacy SDF block-coord caches on demand."""
        if self._sdf_block_coords_cache is not None and self._sdf_index2blocks_cache is not None:
            return
        # Local import keeps the deprecated module out of the normal load path.
        from ..geometry._deprecated_sdf_block_coords import (  # noqa: PLC0415
            build_legacy_sdf_block_arrays,
        )

        subgrid_size = 8
        if self._texture_sdf_data is not None and len(self._texture_sdf_data) > 0:
            subgrid_size = int(self._texture_sdf_data.numpy()[0]["subgrid_size"])
        block_coords, index2blocks = build_legacy_sdf_block_arrays(
            self._texture_sdf_coarse_textures,
            subgrid_size=subgrid_size,
            device=self.device,
        )
        self._sdf_block_coords_cache = block_coords
        self._sdf_index2blocks_cache = index2blocks

    @property
    def has_heightfields(self) -> bool:
        """Deprecated boolean alias for :attr:`heightfield_count`.

        .. deprecated:: 1.3
            Use :attr:`heightfield_count`, or ``heightfield_count > 0`` for
            boolean checks, instead.
        """
        import warnings  # noqa: PLC0415

        warnings.warn(_HAS_HEIGHTFIELDS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.heightfield_count > 0

    @has_heightfields.setter
    def has_heightfields(self, value: bool) -> None:
        import warnings  # noqa: PLC0415

        warnings.warn(_HAS_HEIGHTFIELDS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.heightfield_count = 1 if value else 0

    @property
    def joint_target_q_start(self) -> wp.array | None:
        """Per-joint start index into :attr:`joint_target_q`, shape
        ``(joint_count + 1,)``. Aliases :attr:`joint_q_start` under coord
        layout, :attr:`joint_qd_start` otherwise. Solvers and actuators should
        index :attr:`joint_target_q` through this regardless of layout.
        """
        return self.joint_q_start if self.use_coord_layout_targets else self.joint_qd_start

    @property
    def joint_target_pos(self) -> wp.array | None:
        """Deprecated alias for :attr:`joint_target_q` (DOF-shape only).
        Raises :class:`AttributeError` when this Model was built under
        :attr:`use_coord_layout_targets` ``True``.

        .. deprecated:: 1.3
            Use :attr:`joint_target_q` instead.
        """
        import warnings  # noqa: PLC0415

        from .control import _JOINT_TARGET_POS_DEPRECATION_MSG, _JOINT_TARGET_POS_UNAVAILABLE_MSG  # noqa: PLC0415

        if self.use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_POS_UNAVAILABLE_MSG.replace("Control.", "Model."))
        warnings.warn(
            _JOINT_TARGET_POS_DEPRECATION_MSG.replace("Control.", "Model."),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.joint_target_q

    @joint_target_pos.setter
    def joint_target_pos(self, value: wp.array | None) -> None:
        import warnings  # noqa: PLC0415

        from .control import _JOINT_TARGET_POS_DEPRECATION_MSG, _JOINT_TARGET_POS_UNAVAILABLE_MSG  # noqa: PLC0415

        if self.use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_POS_UNAVAILABLE_MSG.replace("Control.", "Model."))
        warnings.warn(
            _JOINT_TARGET_POS_DEPRECATION_MSG.replace("Control.", "Model."),
            DeprecationWarning,
            stacklevel=2,
        )
        self.joint_target_q = value

    @property
    def joint_target_vel(self) -> wp.array | None:
        """Deprecated alias for :attr:`joint_target_qd`. Raises
        :class:`AttributeError` when this Model was built under
        :attr:`use_coord_layout_targets` ``True``.

        .. deprecated:: 1.3
            Use :attr:`joint_target_qd` instead.
        """
        import warnings  # noqa: PLC0415

        from .control import _JOINT_TARGET_VEL_DEPRECATION_MSG, _JOINT_TARGET_VEL_UNAVAILABLE_MSG  # noqa: PLC0415

        if self.use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_VEL_UNAVAILABLE_MSG.replace("Control.", "Model."))
        warnings.warn(
            _JOINT_TARGET_VEL_DEPRECATION_MSG.replace("Control.", "Model."),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.joint_target_qd

    @joint_target_vel.setter
    def joint_target_vel(self, value: wp.array | None) -> None:
        import warnings  # noqa: PLC0415

        from .control import _JOINT_TARGET_VEL_DEPRECATION_MSG, _JOINT_TARGET_VEL_UNAVAILABLE_MSG  # noqa: PLC0415

        if self.use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_VEL_UNAVAILABLE_MSG.replace("Control.", "Model."))
        warnings.warn(
            _JOINT_TARGET_VEL_DEPRECATION_MSG.replace("Control.", "Model."),
            DeprecationWarning,
            stacklevel=2,
        )
        self.joint_target_qd = value

    def bvh_build_shapes(self, state: State, *, bvh_constructor: str | None = None) -> None:
        """Build or rebuild the shape BVH stored on this model.

        Allocates :attr:`bvh_shapes` and related fields from the current
        shape data and *state*. :meth:`ModelBuilder.finalize` calls this for
        the initial model state. Call it again to rebuild with a custom
        ``bvh_constructor`` or after structural changes. For ordinary state
        changes, use :meth:`bvh_refit_shapes`.

        Args:
            state: Current simulation state with body transforms.
            bvh_constructor: Warp BVH construction algorithm. Valid choices
                are ``"sah"``, ``"median"``, ``"lbvh"``, or ``None`` to use
                Warp's device-dependent default.
        """
        from ..geometry.bvh import (  # noqa: PLC0415
            compute_bvh_group_roots,
            compute_enabled_shapes,
            compute_shape_bvh_bounds_launch,
            compute_shape_local_bounds,
            compute_shape_world_transforms_launch,
        )

        if self.shape_count == 0:
            return

        device = self.device
        shape_count = self.shape_count
        world_count_total = self.world_count + 1

        self.bvh_shape_bounds = wp.empty((shape_count, 2), dtype=wp.vec3f, ndim=2, device=device)
        wp.launch(
            kernel=compute_shape_local_bounds,
            dim=shape_count,
            inputs=[
                self.shape_type,
                self.shape_source_ptr,
                self.gaussians_data,
                self.bvh_shape_bounds,
            ],
            device=device,
        )

        self.bvh_shape_enabled = wp.empty(shape_count, dtype=wp.uint32, device=device)
        num_enabled = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernel=compute_enabled_shapes,
            dim=shape_count,
            inputs=[
                self.shape_type,
                self.shape_flags,
                self.bvh_shape_enabled,
                num_enabled,
            ],
            device=device,
        )
        self.bvh_shape_count_enabled = int(num_enabled.numpy()[0])
        self.bvh_shape_world_transforms = wp.empty(shape_count, dtype=wp.transformf, device=device)

        if self.bvh_shape_count_enabled == 0:
            return

        compute_shape_world_transforms_launch(self, state)

        lowers = wp.zeros(self.bvh_shape_count_enabled, dtype=wp.vec3f, device=device)
        uppers = wp.zeros(self.bvh_shape_count_enabled, dtype=wp.vec3f, device=device)
        groups = wp.zeros(self.bvh_shape_count_enabled, dtype=wp.int32, device=device)
        compute_shape_bvh_bounds_launch(self, lowers, uppers, groups)
        self.bvh_shapes = wp.Bvh(lowers, uppers, constructor=bvh_constructor, groups=groups)

        self.bvh_shapes_group_roots = wp.zeros(world_count_total, dtype=wp.int32, device=device)
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=world_count_total,
            inputs=[self.bvh_shapes.id, self.bvh_shapes_group_roots],
            device=device,
        )

    def bvh_refit_shapes(self, state: State) -> None:
        """Refit the shape BVH stored on this model for the current state.

        The shape BVH is built automatically by :meth:`ModelBuilder.finalize`.
        Manually populated models must call :meth:`bvh_build_shapes` first.
        Updates world-space shape transforms from ``state.body_q`` and refits
        the BVH in place.

        Args:
            state: Current simulation state with body transforms.
        """
        from ..geometry.bvh import (  # noqa: PLC0415
            compute_shape_bvh_bounds_launch,
            compute_shape_world_transforms_launch,
        )

        if self.shape_count == 0:
            return
        if self.bvh_shape_enabled is None:
            raise RuntimeError("Model.bvh_refit_shapes() requires Model.bvh_build_shapes() to have been called first.")
        if self.bvh_shape_count_enabled == 0:
            return
        if self.bvh_shapes is None:
            raise RuntimeError("Model.bvh_refit_shapes() requires Model.bvh_build_shapes() to have been called first.")

        compute_shape_world_transforms_launch(self, state)
        compute_shape_bvh_bounds_launch(self, self.bvh_shapes.lowers, self.bvh_shapes.uppers, self.bvh_shapes.groups)
        self.bvh_shapes.refit()

    def bvh_build_particles(self, state: State, *, bvh_constructor: str | None = None) -> None:
        """Build or rebuild the particle BVH stored on this model.

        Allocates :attr:`bvh_particles` and related fields from particle data
        in *state*. :meth:`ModelBuilder.finalize` calls this for the initial
        model state when particles are present. Call it again to rebuild with
        a custom ``bvh_constructor``. For ordinary state changes, use
        :meth:`bvh_refit_particles`.

        Args:
            state: Current simulation state with particle positions.
            bvh_constructor: Warp BVH construction algorithm. Valid choices
                are ``"sah"``, ``"median"``, ``"lbvh"``, or ``None`` to use
                Warp's device-dependent default.
        """
        from ..geometry.bvh import compute_bvh_group_roots, compute_particle_bvh_bounds_launch  # noqa: PLC0415

        if state.particle_q is None or state.particle_count == 0:
            return

        device = self.device
        world_count_total = self.world_count + 1
        num_particles = state.particle_count

        lowers = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
        uppers = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
        groups = wp.zeros(num_particles, dtype=wp.int32, device=device)
        compute_particle_bvh_bounds_launch(self, state, lowers, uppers, groups)
        self.bvh_particles = wp.Bvh(lowers, uppers, constructor=bvh_constructor, groups=groups)

        self.bvh_particles_group_roots = wp.zeros(world_count_total, dtype=wp.int32, device=device)
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=world_count_total,
            inputs=[self.bvh_particles.id, self.bvh_particles_group_roots],
            device=device,
        )

    def bvh_refit_particles(self, state: State) -> None:
        """Refit the particle BVH stored on this model for the current state.

        The particle BVH is built automatically by :meth:`ModelBuilder.finalize`
        when particles are present. Manually populated models must call
        :meth:`bvh_build_particles` first.
        Recomputes particle bounds from ``state.particle_q`` and refits the
        BVH in place.

        Args:
            state: Current simulation state with particle positions.
        """
        from ..geometry.bvh import compute_particle_bvh_bounds_launch  # noqa: PLC0415

        if state.particle_q is None or state.particle_count == 0:
            return
        if self.bvh_particles is None:
            raise RuntimeError(
                "Model.bvh_refit_particles() requires Model.bvh_build_particles() to have been called first."
            )

        compute_particle_bvh_bounds_launch(
            self, state, self.bvh_particles.lowers, self.bvh_particles.uppers, self.bvh_particles.groups
        )
        self.bvh_particles.refit()

    def state(self, requires_grad: bool | None = None) -> State:
        """
        Create and return a new :class:`State` object for this model.

        The returned state is initialized with the initial configuration from the model description.

        Args:
            requires_grad: Whether the state variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.

        Returns:
            The state object.
        """

        requested = self.get_requested_state_attributes()

        s = State()
        if requires_grad is None:
            requires_grad = self.requires_grad

        # particles
        if self.particle_count:
            s.particle_q = wp.clone(self.particle_q, requires_grad=requires_grad)
            s.particle_qd = wp.clone(self.particle_qd, requires_grad=requires_grad)
            s.particle_f = wp.zeros_like(self.particle_qd, requires_grad=requires_grad)

        # rigid bodies
        if self.body_count:
            s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
            s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
            s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        # joints
        if self.joint_count:
            s.joint_q = wp.clone(self.joint_q, requires_grad=requires_grad)
            s.joint_qd = wp.clone(self.joint_qd, requires_grad=requires_grad)

        self._add_requested_state_attributes(s, requested, requires_grad=requires_grad)

        # attach custom attributes with assignment==STATE
        self._add_custom_attributes(s, Model.AttributeAssignment.STATE, requires_grad=requires_grad)

        return s

    def _add_requested_state_attributes(
        self,
        state: State,
        requested: list[str],
        requires_grad: bool = False,
    ) -> None:
        """Allocate optional built-in state attributes requested by name."""
        for full_name in requested:
            template = State.EXTENDED_ATTRIBUTE_TEMPLATES.get(full_name)
            if template is None:
                continue

            frequency = getattr(Model.AttributeFrequency, template.frequency)
            value = wp.zeros(
                self._attribute_frequency_count(frequency),
                dtype=template.dtype,
                device=self.device,
                requires_grad=requires_grad,
            )
            if ":" in full_name:
                namespace_name, attr_name = full_name.split(":", 1)
                namespace = getattr(state, namespace_name, None)
                if namespace is None:
                    namespace = Model.AttributeNamespace(namespace_name)
                    setattr(state, namespace_name, namespace)
                setattr(namespace, attr_name, value)
            else:
                setattr(state, full_name, value)

    def _attribute_frequency_count(self, frequency: Model.AttributeFrequency | str) -> int:
        if isinstance(frequency, str):
            return int(self.custom_frequency_counts[frequency])

        if frequency == Model.AttributeFrequency.ONCE:
            return 1
        count_attr = Model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS.get(frequency)
        if count_attr is None:
            raise ValueError(f"Unsupported attribute frequency: {frequency!r}")
        return int(getattr(self, count_attr))

    def control(self, requires_grad: bool | None = None, clone_variables: bool = True) -> Control:
        """
        Create and return a new :class:`Control` object for this model.

        The returned control object is initialized with the control inputs from the model description.

        Args:
            requires_grad: Whether the control variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.
            clone_variables: If True, clone the control input arrays; if False, use references.

        Returns:
            The initialized control object.
        """
        c = Control()
        c._use_coord_layout_targets = self.use_coord_layout_targets
        if requires_grad is None:
            requires_grad = self.requires_grad
        if clone_variables:
            if self.joint_count:
                if self.joint_target_q is not None:
                    c.joint_target_q = wp.clone(self.joint_target_q, requires_grad=requires_grad)
                if self.joint_target_qd is not None:
                    c.joint_target_qd = wp.clone(self.joint_target_qd, requires_grad=requires_grad)
                c.joint_act = wp.clone(self.joint_act, requires_grad=requires_grad)
                c.joint_f = wp.clone(self.joint_f, requires_grad=requires_grad)
            if self.tri_count:
                c.tri_activations = wp.clone(self.tri_activations, requires_grad=requires_grad)
            if self.tet_count:
                c.tet_activations = wp.clone(self.tet_activations, requires_grad=requires_grad)
            if self.muscle_count:
                c.muscle_activations = wp.clone(self.muscle_activations, requires_grad=requires_grad)
        else:
            c.joint_target_q = self.joint_target_q
            c.joint_target_qd = self.joint_target_qd
            c.joint_act = self.joint_act
            c.joint_f = self.joint_f
            c.tri_activations = self.tri_activations
            c.tet_activations = self.tet_activations
            c.muscle_activations = self.muscle_activations
        # attach custom attributes with assignment==CONTROL
        self._add_custom_attributes(
            c, Model.AttributeAssignment.CONTROL, requires_grad=requires_grad, clone_arrays=clone_variables
        )
        return c

    def set_gravity(
        self,
        gravity: tuple[float, float, float] | list | wp.vec3 | np.ndarray,
        world: int | None = None,
    ) -> None:
        """
        Set gravity for runtime modification.

        Args:
            gravity: Gravity vector (3,) or per-world array (world_count, 3).
            world: If provided, set gravity only for this world.

        Note:
            Call ``solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)`` after.

            Global entities (particles/bodies not assigned to a specific world) use
            gravity from world 0.
        """
        gravity_np = np.asarray(gravity, dtype=np.float32)

        if world is not None:
            if gravity_np.shape != (3,):
                raise ValueError("Expected single gravity vector (3,) when world is specified")
            if world < 0 or world >= self.world_count:
                raise IndexError(f"world {world} out of range [0, {self.world_count})")
            current = self.gravity.numpy()
            current[world] = gravity_np
            self.gravity.assign(current)
        elif gravity_np.ndim == 1:
            self.gravity.fill_(gravity_np)
        else:
            if len(gravity_np) != self.world_count:
                raise ValueError(f"Expected {self.world_count} gravity vectors, got {len(gravity_np)}")
            self.gravity.assign(gravity_np)

    def _init_collision_pipeline(self, enable_rigid_soft_full_surface_contact: bool = False):
        """
        Initialize a :class:`CollisionPipeline` for this model.

        This method creates a default collision pipeline for the model. The pipeline is cached on
        the model for subsequent use by :meth:`collide`.

        Args:
            enable_rigid_soft_full_surface_contact: Size the soft-contact buffer for the full-surface
                EDGE/FACE passes (see :meth:`collide`).
        """
        from .collide import CollisionPipeline  # noqa: PLC0415

        self._collision_pipeline = CollisionPipeline(
            self,
            broad_phase="explicit",
            enable_rigid_soft_full_surface_contact=enable_rigid_soft_full_surface_contact,
        )

    def contacts(
        self: Model,
        collision_pipeline: CollisionPipeline | None = None,
    ) -> Contacts:
        """
        Create and return a :class:`Contacts` object for this model.

        .. deprecated:: 1.5

            Create a :class:`CollisionPipeline` and call
            :meth:`CollisionPipeline.contacts` instead.

        This method initializes a collision pipeline with default arguments (when not already
        cached) and allocates a contacts buffer suitable for storing collision detection results.
        Call :meth:`collide` to run the collision detection and populate the contacts object.

        Note:
            Rigid contact gaps are controlled per-shape via :attr:`Model.shape_gap`, which is populated
            from ``ModelBuilder.ShapeConfig.gap`` [m] during model building. If a shape doesn't specify a gap [m],
            it defaults to ``builder.rigid_gap`` [m]. To adjust contact gaps [m], set them before calling
            :meth:`ModelBuilder.finalize`.
        Returns:
            The contact object containing collision information.
        """
        warnings.warn(
            "Model.contacts() is deprecated; create a CollisionPipeline and call pipeline.contacts() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if collision_pipeline is not None:
            self._collision_pipeline = collision_pipeline
        if self._collision_pipeline is None:
            self._init_collision_pipeline()

        return self._collision_pipeline.contacts()

    def collide(
        self,
        state: State,
        contacts: Contacts | None = None,
        *,
        collision_pipeline: CollisionPipeline | None = None,
        enable_rigid_soft_full_surface_contact: bool = False,
    ) -> Contacts:
        """
        Generate contact points for the particles and rigid bodies in the model using the default collision
        pipeline.

        .. deprecated:: 1.5

            Create a :class:`CollisionPipeline` and call
            :meth:`CollisionPipeline.collide` instead.

        Args:
            state: The current simulation state.
            contacts: The contacts buffer to populate (will be cleared first). If None, a new
                contacts buffer is allocated via :meth:`contacts`.
            collision_pipeline: Optional collision pipeline override.
            enable_rigid_soft_full_surface_contact: When ``True``, additionally run the triangle-driven
                soft EDGE/FACE passes that detect soft edge / face vs rigid contacts the per-particle
                SDF path misses, written into the E/F ranges of ``Contacts.soft_contact_*``. Default
                ``False`` reproduces the per-particle behaviour bit-for-bit. This flag is applied when
                the collision pipeline is allocated (its soft-contact buffer must be sized for the extra
                records), so it takes effect only on the first ``collide()``/``contacts()`` call that
                creates the pipeline. Passing ``True`` once a pipeline sized without it is cached raises
                ``ValueError``. Participating mesh/convex shapes must also have volume SDFs provisioned via
                :meth:`ModelBuilder.ShapeConfig.configure_sdf` (e.g. ``configure_sdf(force_sdf=True)`` on
                ``default_shape_cfg``) before finalize, or pipeline construction raises.
        """
        warnings.warn(
            "Model.collide() is deprecated; create a CollisionPipeline and call "
            "pipeline.collide(state, contacts) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if collision_pipeline is not None:
            self._collision_pipeline = collision_pipeline
        if self._collision_pipeline is None:
            self._init_collision_pipeline(enable_rigid_soft_full_surface_contact=enable_rigid_soft_full_surface_contact)
        elif (
            enable_rigid_soft_full_surface_contact
            and not self._collision_pipeline.enable_rigid_soft_full_surface_contact
        ):
            raise ValueError(
                "enable_rigid_soft_full_surface_contact=True requires a collision pipeline initialized with "
                "the flag so its soft-contact buffer is sized for the edge/face passes, but the cached "
                "pipeline was built with it disabled. Pass a fresh collision_pipeline=, or enable the flag "
                "on the first collide()/contacts() call that allocates the pipeline."
            )

        if contacts is None:
            contacts = self._collision_pipeline.contacts()

        self._collision_pipeline.collide(state, contacts)
        return contacts

    def request_state_attributes(self, *attributes: str) -> None:
        """
        Request that specific state attributes be allocated when creating a State object.

        See :ref:`extended_state_attributes` for details and usage.

        Args:
            *attributes: Variable number of attribute names (strings).
        """
        State.validate_extended_attributes(attributes)
        self._requested_state_attributes.update(attributes)

    def request_contact_attributes(self, *attributes: str) -> None:
        """
        Request that specific contact attributes be allocated when creating a Contacts object.

        Args:
            *attributes: Variable number of attribute names (strings).
        """
        Contacts.validate_extended_attributes(attributes)
        self._requested_contact_attributes.update(attributes)

    def get_requested_contact_attributes(self) -> set[str]:
        """
        Get the set of requested contact attribute names.

        Returns:
            The set of requested contact attributes.
        """
        return self._requested_contact_attributes

    def _add_custom_attributes(
        self,
        destination: object,
        assignment: Model.AttributeAssignment,
        requires_grad: bool = False,
        clone_arrays: bool = True,
    ) -> None:
        """
        Add custom attributes of a specific assignment type to a destination object.

        Args:
            destination: The object to add attributes to (State, Control, or Contacts)
            assignment: The assignment type to filter attributes by
            requires_grad: Whether cloned arrays should have requires_grad enabled
            clone_arrays: Whether to clone wp.arrays (True) or use references (False)
        """
        for full_name, spec in self._iter_attribute_specs():
            attribute_assignment = Model.AttributeAssignment.MODEL if spec.assignment is None else spec.assignment
            if attribute_assignment != assignment:
                continue

            # Parse namespace from full_name (format: "namespace:attr_name" or "attr_name")
            if ":" in full_name:
                namespace, attr_name = full_name.split(":", 1)
                # Get source from namespaced location on model
                ns_obj = getattr(self, namespace, None)
                if ns_obj is None:
                    raise AttributeError(f"Namespace '{namespace}' does not exist on the model")
                src = getattr(ns_obj, attr_name, None)
                if src is None:
                    raise AttributeError(
                        f"Attribute '{namespace}.{attr_name}' is registered but does not exist on the model"
                    )
                # Create namespace on destination if it doesn't exist
                if not hasattr(destination, namespace):
                    setattr(destination, namespace, Model.AttributeNamespace(namespace))
                dest = getattr(destination, namespace)
            else:
                # Non-namespaced attribute - add directly to destination
                attr_name = full_name
                src = getattr(self, attr_name, None)
                if src is None:
                    raise AttributeError(
                        f"Attribute '{attr_name}' is registered in attribute_frequency but does not exist on the model"
                    )
                dest = destination

            # Add attribute to the determined destination (either destination or dest_ns)
            if isinstance(src, wp.array):
                if clone_arrays:
                    setattr(dest, attr_name, wp.clone(src, requires_grad=requires_grad))
                else:
                    setattr(dest, attr_name, src)
            else:
                setattr(dest, attr_name, src)

    def add_attribute(
        self,
        name: str,
        attrib: wp.array | list[Any],
        frequency: Model.AttributeFrequency | str,
        assignment: Model.AttributeAssignment | None = None,
        namespace: str | None = None,
        references: str | None = None,
    ):
        """
        Add a custom attribute to the model.

        Args:
            name: Name of the attribute.
            attrib: The array to add as an attribute. Can be a wp.array for
                numeric types or a list for string attributes.
            frequency: The frequency of the attribute.
                Can be a Model.AttributeFrequency enum value or a string for custom frequencies.
            assignment: The assignment category using Model.AttributeAssignment enum.
                Determines which object will hold the attribute.
            namespace: Namespace for the attribute.
                If None, attribute is added directly to the assignment object (e.g., model.attr_name).
                If specified, attribute is added to a namespace object (e.g., model.namespace_name.attr_name).
            references: Entity or custom-frequency domain indexed by the
                attribute values, or ``None`` when values are not references.

        Raises:
            AttributeError: If the attribute already exists or is on the wrong device.
        """
        if isinstance(attrib, wp.array) and attrib.device != self.device:
            raise AttributeError(f"Attribute '{name}' device mismatch (model={self.device}, got={attrib.device})")

        # Handle namespaced attributes
        if namespace:
            # Create namespace object if it doesn't exist
            if not hasattr(self, namespace):
                setattr(self, namespace, Model.AttributeNamespace(namespace))

            ns_obj = getattr(self, namespace)
            if name in ns_obj.__dict__ or name in ns_obj._deprecated_aliases:
                raise AttributeError(f"Attribute already exists: {namespace}.{name}")

            setattr(ns_obj, name, attrib)
            full_name = f"{namespace}:{name}"
        else:
            # Add directly to model
            if hasattr(self, name):
                raise AttributeError(f"Attribute already exists: {name}")
            setattr(self, name, attrib)
            full_name = name

        reference_frequency = self._normalize_attribute_reference(references)
        if reference_frequency is not None and isinstance(attrib, wp.array):
            integral_dtypes = (wp.int8, wp.int16, wp.int32, wp.int64, wp.uint8, wp.uint16, wp.uint32, wp.uint64)
            scalar_dtype = getattr(attrib.dtype, "_wp_scalar_type_", attrib.dtype)
            if scalar_dtype not in integral_dtypes or attrib.ndim != 1:
                raise ValueError(
                    f"Reference attribute '{full_name}' must be a 1-D array with integral components, "
                    f"got dtype={attrib.dtype}, ndim={attrib.ndim}"
                )
        self._set_attribute_spec(
            full_name,
            Model.AttributeSpec(
                frequency=frequency,
                assignment=assignment,
                references=reference_frequency,
            ),
        )

    def get_attribute_frequency(self, name: str) -> Model.AttributeFrequency | str:
        """
        Get the frequency of an attribute.

        Args:
            name: Name of the attribute.

        Returns:
            The frequency of the attribute.
                Either a Model.AttributeFrequency enum value or a string for custom frequencies.

        Raises:
            KeyError: If the attribute frequency is not known.
        """
        spec = self._attribute_spec(name)
        if spec is None:
            raise KeyError(f"Attribute frequency of '{name}' is not known")
        return spec.frequency

    def get_custom_frequency_count(self, frequency: str) -> int:
        """
        Get the count for a custom frequency.

        Args:
            frequency: The custom frequency (e.g., ``"mujoco:pair"``).

        Returns:
            The count of elements with this frequency.

        Raises:
            KeyError: If the frequency is not known.
        """
        if frequency not in self.custom_frequency_counts:
            raise KeyError(f"Custom frequency '{frequency}' is not known")
        return self.custom_frequency_counts[frequency]

    def get_requested_state_attributes(self) -> list[str]:
        """
        Get the list of requested state attribute names that have been requested on the model.

        See :ref:`extended_state_attributes` for details.

        Returns:
            The list of requested state attributes.
        """
        attributes = []

        if self.particle_count:
            attributes.extend(
                (
                    "particle_q",
                    "particle_qd",
                    "particle_f",
                )
            )
        if self.body_count:
            attributes.extend(
                (
                    "body_q",
                    "body_qd",
                    "body_f",
                )
            )
        if self.joint_count:
            attributes.extend(("joint_q", "joint_qd"))

        attributes.extend(self._requested_state_attributes.difference(attributes))
        return attributes
