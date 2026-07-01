# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Geometry Model Types & Containers
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import warp as wp

# TODO: from .....sim.builder import ModelBuilder
from .....core.types import override
from .....geometry.flags import ShapeFlags
from .shapes import ShapeDescriptorType
from .types import Descriptor

if TYPE_CHECKING:
    from .....utils.heightfield import HeightfieldData

###
# Module interface
###

__all__ = [
    "GeometriesData",
    "GeometriesModel",
    "GeometryDescriptor",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Base Geometry Containers
###


@dataclass
class GeometryDescriptor(Descriptor):
    """
    A container to describe a geometry entity.

    A geometry entity is an abstraction to represent the composition
    of a shape, with a pose w.r.t the world frame of a scene. Each
    geometry descriptor bundles the unique object identifiers of the
    entity, indices to the associated body, the offset pose w.r.t.
    the body, and a shape descriptor.
    """

    ###
    # Basic Attributes
    ###

    body: int = -1
    """
    Index of the body to which the geometry entity is attached.
    Defaults to `-1`, indicating that the geometry has not yet been assigned to a body.
    The value `-1` also indicates that the geometry, by default, is statically attached to the world.
    """

    shape: ShapeDescriptorType | None = None
    """Definition of the shape of the geometry entity of type :class:`ShapeDescriptorType`."""

    offset: wp.transformf = field(default_factory=wp.transform_identity)
    """Offset pose of the geometry entity w.r.t. its corresponding body, of type :class:`wp.transformf`."""

    # TODO: Use Model.ShapeConfig instead of all these individual fields
    # config: ModelBuilder.ShapeConfig = field(default_factory=ModelBuilder.ShapeConfig)

    ###
    # Collision Attributes
    ###

    material: str | int | None = None
    """
    The material assigned to the collision geometry instance.
    Can be specified either as a string name or an integer index.
    Defaults to `None`, indicating the default material.
    """

    group: int = 1
    """
    The collision group assigned to the collision geometry.
    Defaults to the default group with value `1`.
    """

    collides: int = 1
    """
    The collision groups with which the collision geometry can collide.
    Defaults to enabling collisions with the default group with value `1`.
    """

    max_contacts: int = 0
    """
    The maximum number of contacts to generate for the collision geometry.
    This value provides a hint to the model builder when allocating memory for contacts.
    Defaults to `0`, indicating no limit is imposed on the number of contacts generated for this geometry.
    """

    gap: float = 0.0
    """
    Additional detection threshold [m] for this geometry.

    Pairwise effect is additive (``g_a + g_b``): the broadphase expands each shape's
    bounding volume by ``margin + gap``, and the narrowphase keeps a contact when
    ``d <= gap_a + gap_b``(with ``d`` measured relative to margin-shifted surfaces).

    Defaults to `0.0`.
    """

    margin: float = 0.0
    """
    Surface offset [m] for this geometry.

    Pairwise effect is additive (``m_a + m_b``): contacts are
    evaluated against the signed distance to the margin-shifted
    surfaces, so resting separation equals ``m_a + m_b``.

    Defaults to `0.0`.
    """

    ###
    # Metadata - to be set by the model builder when added
    ###

    wid: int = -1
    """
    Index of the world to which the body belongs.
    Defaults to `-1`, indicating that the body has not yet been added to a world.
    """

    gid: int = -1
    """
    Index of the geometry w.r.t. its world.
    Defaults to `-1`, indicating that the geometry has not yet been added to a world.
    """

    mid: int = -1
    """
    The material index assigned to the collision geometry.
    Defaults to `-1` indicating that the default material will be assigned.
    """

    flags: int = ShapeFlags.VISIBLE | ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES
    """
    Shape flags of the geometry entity, used to specify additional properties of the geometry.
    Defaults to `ShapeFlags.VISIBLE | ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES`,
    indicating that the geometry is visible and can collide with shapes and particles.
    """

    ###
    # Operations
    ###

    @property
    def is_collidable(self) -> bool:
        """Returns `True` if the geometry is collidable (i.e., group > 0)."""
        return self.group > 0

    @override
    def __hash__(self):
        """Returns a hash computed using the shape descriptor's hash implementation."""
        # NOTE: The name-uid-based hash implementation is called if no shape is defined
        if self.shape is None:
            return super().__hash__()
        # Otherwise, use the shape's hash implementation
        return self.shape.__hash__()

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the GeometryDescriptor."""
        return (
            f"GeometryDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"body: {self.body},\n"
            f"shape: {self.shape},\n"
            f"offset: {self.offset},\n"
            f"material: {self.material},\n"
            f"group: {self.group},\n"
            f"collides: {self.collides},\n"
            f"max_contacts: {self.max_contacts}\n"
            f"gap: {self.gap},\n"
            f"margin: {self.margin},\n"
            f"wid: {self.wid},\n"
            f"gid: {self.gid},\n"
            f"mid: {self.mid},\n"
            f")"
        )

    @staticmethod
    def copy_without_shape(geom: GeometryDescriptor) -> GeometryDescriptor:
        """Returns a copy of a descriptor, but with the shape field set to None"""
        return replace(geom, shape=None)


@dataclass
class GeometriesModel:
    """
    An SoA-based container to hold time-invariant model data of a set of generic geometry elements.
    """

    ###
    # Meta-Data
    ###

    num_geoms: int = 0
    """Total number of geometry entities in the model (host-side)."""

    num_collidable: int = 0
    """Total number of collidable geometry entities in the model (host-side)."""

    num_collidable_pairs: int = 0
    """Total number of collidable geometry pairs in the model (host-side)."""

    num_excluded_pairs: int = 0
    """Total number of excluded geometry pairs in the model (host-side)."""

    model_minimum_contacts: int = 0
    """The minimum number of contacts required for the entire model (host-side)."""

    world_minimum_contacts: list[int] | None = None
    """
    List of the minimum number of contacts required for each world in the model (host-side).
    The sum of all elements in `world_minimum_contacts` should equal `model_minimum_contacts`.
    """

    label: list[str] | None = None
    """
    A list containing the label of each geometry.
    Length of ``num_geoms``.
    """

    ###
    # Identifiers
    ###

    wid: wp.array[wp.int32] | None = None
    """
    World index of each geometry entity.
    Shape of ``(num_geoms,)``.
    """

    gid: wp.array[wp.int32] | None = None
    """
    Geometry index of each geometry entity w.r.t its world.
    Shape of ``(num_geoms,)``.
    """

    bid: wp.array[wp.int32] | None = None
    """
    Body index of each geometry entity.
    Shape of ``(num_geoms,)``.
    """

    ###
    # Parameterization
    ###

    type: wp.array[wp.int32] | None = None
    """
    Shape index of each geometry entity.
    Shape of ``(num_geoms,)``.
    """

    flags: wp.array[wp.int32] | None = None
    """
    Shape flags of each geometry entity.
    Shape of ``(num_geoms,)``.
    """

    ptr: wp.array[wp.uint64] | None = None
    """
    Pointer to the source data of the shape.
    For primitive shapes this is `0` indicating NULL, otherwise it points to
    the shape data, which can correspond to a mesh, heightfield, or SDF.
    Shape of ``(num_geoms,)``.
    """

    params: wp.array[wp.vec3f] | None = None
    """
    Shape parameters of each geometry entity if they are shape primitives.
    Shape of ``(num_geoms,)``.
    """

    offset: wp.array[wp.transformf] | None = None
    """
    Offset poses of the geometry elements w.r.t. their corresponding bodies.
    Shape of ``(num_geoms,)``.
    """

    ###
    # Collisions
    ###

    material: wp.array[wp.int32] | None = None
    """
    Material index assigned to each collision geometry.
    Shape of ``(num_geoms,)``.
    """

    group: wp.array[wp.int32] | None = None
    """
    Collision group assigned to each collision geometry.
    Shape of ``(num_geoms,)``.
    """

    gap: wp.array[wp.float32] | None = None
    """
    Additional detection threshold [m] for each collision geometry.
    Pairwise additive.  Used by both broadphase (AABB expansion) and
    narrowphase (contact retention).
    Shape of ``(num_geoms,)``.
    """

    margin: wp.array[wp.float32] | None = None
    """
    Surface offset [m] for each collision geometry.
    Pairwise additive.  Determines resting separation between shapes.
    Shape of ``(num_geoms,)``.
    """

    collidable_pairs: wp.array[wp.vec2i] | None = None
    """
    Geometry-pair indices that are explicitly considered for collision detection.
    This array is used in broad-phase collision detection.
    Shape of ``(num_collidable_pairs,)``.
    """

    excluded_pairs: wp.array[wp.vec2i] | None = None
    """
    Geometry-pair indices that are explicitly excluded from collision detection.
    This array is used in broad-phase collision detection.
    Shape of ``(num_excluded_geom_pairs,)``.
    """

    ###
    # Mesh / Heightfield Data
    ###

    heightfield_index: wp.array[wp.int32] | None = None
    """Per-shape heightfield index (``-1`` for non-heightfield shapes)."""

    heightfield_data: wp.array[HeightfieldData] | None = None
    """Concatenated :class:`HeightfieldData` structs for all heightfields."""

    heightfield_elevations: wp.array[wp.float32] | None = None
    """Concatenated elevation samples for all heightfields."""

    collision_aabb_lower: wp.array[wp.vec3f] | None = None
    """Per-shape local-space collision AABB lower bounds."""

    collision_aabb_upper: wp.array[wp.vec3f] | None = None
    """Per-shape local-space collision AABB upper bounds."""

    collision_radius: wp.array[wp.float32] | None = None
    """Per-shape bounding-sphere radius for broadphase AABB computation."""

    voxel_resolution: wp.array[wp.vec3i] | None = None
    """Per-shape voxel resolution for mesh contact reduction."""


@dataclass
class GeometriesData:
    """
    An SoA-based container to hold time-varying data of a set of generic geometry entities.

    Attributes:
        num_geoms: The total number of geometry entities in the model (host-side).
        pose: The poses of the geometry entities in world coordinates.
            Shape of ``(num_geoms,)``.
    """

    num_geoms: int = 0
    """Total number of geometry entities in the model (host-side)."""

    pose: wp.array[wp.transformf] | None = None
    """
    The poses of the geometry entities in world coordinates.
    Shape of ``(num_geoms,)``.
    """


###
# Kernels
###


@wp.kernel
def _update_geometries_state(
    # Inputs:
    geom_bid: wp.array[wp.int32],
    geom_offset: wp.array[wp.transformf],
    body_pose: wp.array[wp.transformf],
    # Outputs:
    geom_pose: wp.array[wp.transformf],
):
    """
    A kernel to update poses of geometry entities in world
    coordinates from the poses of their associated bodies.

    Inputs:
        geom_bid: Array of per-geom body indices.
            Shape of ``(num_geoms,)``.
        geom_offset: Array of per-geom pose offsets w.r.t. their associated bodies.
            Shape of ``(num_geoms,)``.
        body_pose: Array of per-body poses in world coordinates.
            Shape of ``(num_bodies,)``.

    Outputs:
        geom_pose: Array of per-geom poses in world coordinates.
            Shape of ``(num_geoms,)``.
    """
    # Retrieve the geometry index from the thread grid
    gid = wp.tid()

    # Retrieve the body index associated with the geometry
    bid = geom_bid[gid]

    # Retrieve the pose of the corresponding body
    X_b = wp.transform_identity(dtype=wp.float32)
    if bid > -1:
        X_b = body_pose[bid]

    # Retrieve the geometry offset pose w.r.t. the body
    X_bg = geom_offset[gid]

    # Compute the geometry pose in world coordinates
    X_g = wp.transform_multiply(X_b, X_bg)

    # Store the updated geometry pose
    geom_pose[gid] = X_g


###
# Launchers
###


def update_geometries_state(
    body_poses: wp.array[wp.transformf],
    geom_model: GeometriesModel,
    geom_data: GeometriesData,
):
    """
    Launches a kernel to update poses of geometry entities in
    world coordinates from the poses of their associated bodies.

    Args:
        body_poses: The poses of the bodies in world coordinates.
            Shape of ``(num_bodies,)``.
        geom_model: The model container holding time-invariant geometry data.
        geom_data: The data container of the geometry elements.
    """
    wp.launch(
        _update_geometries_state,
        dim=geom_model.num_geoms,
        inputs=[geom_model.bid, geom_model.offset, body_poses],
        outputs=[geom_data.pose],
        device=body_poses.device,
    )
