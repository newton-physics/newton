# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Geometry Model Types & Containers
"""

from dataclasses import dataclass, field

import warp as wp

# TODO: from .....sim.builder import ModelBuilder
from .....geometry.flags import ShapeFlags
from .shapes import ShapeDescriptorType
from .types import Descriptor, float32, int32, override, transformf

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
    Index of the body to which the geometry entity is attached.\n
    Defaults to `-1`, indicating that the geometry has not yet been assigned to a body.\n
    The value `-1` also indicates that the geometry, by default, is statically attached to the world.
    """

    shape: ShapeDescriptorType | None = None
    """Definition of the shape of the geometry entity of type :class:`ShapeDescriptorType`."""

    offset: transformf = field(default_factory=wp.transform_identity)
    """Offset pose of the geometry entity w.r.t. its corresponding body, of type :class:`transformf`."""

    # TODO: Use Model.ShapeConfig instead of all these individual fields
    # config: ModelBuilder.ShapeConfig = field(default_factory=ModelBuilder.ShapeConfig)

    ###
    # Collision Attributes
    ###

    material: str | int | None = None
    """
    The material assigned to the collision geometry instance.\n
    Can be specified either as a string name or an integer index.\n
    Defaults to `None`, indicating the default material.
    """

    group: int = 1
    """
    The collision group assigned to the collision geometry.\n
    Defaults to the default group with value `1`.
    """

    collides: int = 1
    """
    The collision groups with which the collision geometry can collide.\n
    Defaults to enabling collisions with the default group with value `1`.
    """

    max_contacts: int = 0
    """
    The maximum number of contacts to generate for the collision geometry.\n
    This value provides a hint to the model builder when allocating memory for contacts.\n
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
    Index of the world to which the body belongs.\n
    Defaults to `-1`, indicating that the body has not yet been added to a world.
    """

    gid: int = -1
    """
    Index of the geometry w.r.t. its world.\n
    Defaults to `-1`, indicating that the geometry has not yet been added to a world.
    """

    mid: int = -1
    """
    The material index assigned to the collision geometry.\n
    Defaults to `-1` indicating that the default material will be assigned.
    """

    flags: int = ShapeFlags.VISIBLE | ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES
    """
    Shape flags of the geometry entity, used to specify additional properties of the geometry.\n
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
    List of the minimum number of contacts required for each world in the model (host-side).\n
    The sum of all elements in `world_minimum_contacts` should equal `model_minimum_contacts`.
    """

    label: list[str] | None = None
    """
    A list containing the label of each geometry.\n
    Length of ``num_geoms`` and type :class:`str`.
    """

    ###
    # Identifiers
    ###

    wid: wp.array | None = None
    """
    World index of each geometry entity.\n
    Shape of ``(num_geoms,)`` and type :class:`int32`.
    """

    gid: wp.array | None = None
    """
    Geometry index of each geometry entity w.r.t its world.\n
    Shape of ``(num_geoms,)`` and type :class:`int32`.
    """

    bid: wp.array | None = None
    """
    Body index of each geometry entity.\n
    Shape of ``(num_geoms,)`` and type :class:`int32`.
    """

    ###
    # Parameterization
    ###

    type: wp.array | None = None
    """
    Shape index of each geometry entity.\n
    Shape of ``(num_geoms,)`` and type :class:`int32`.
    """

    flags: wp.array | None = None
    """
    Shape flags of each geometry entity.\n
    Shape of ``(num_geoms,)`` and type :class:`int32`.
    """

    ptr: wp.array | None = None
    """
    Pointer to the source data of the shape.\n
    For primitive shapes this is `0` indicating NULL, otherwise it points to
    the shape data, which can correspond to a mesh, heightfield, or SDF.\n
    Shape of ``(num_geoms,)`` and type :class:`uint64`.
    """

    params: wp.array | None = None
    """
    Shape parameters of each geometry entity if they are shape primitives.\n
    Shape of ``(num_geoms,)`` and type :class:`vec4f`.
    """

    offset: wp.array | None = None
    """
    Offset poses of the geometry elements w.r.t. their corresponding bodies.\n
    Shape of ``(num_geoms,)`` and type :class:`transformf`.
    """

    ###
    # Collisions
    ###

    material: wp.array | None = None
    """
    Material index assigned to each collision geometry.\n
    Shape of ``(num_geoms,)`` and type :class:`int32`.
    """

    group: wp.array | None = None
    """
    Collision group assigned to each collision geometry.\n
    Shape of ``(num_geoms,)`` and type :class:`uint32`.
    """

    gap: wp.array | None = None
    """
    Additional detection threshold [m] for each collision geometry.\n
    Pairwise additive.  Used by both broadphase (AABB expansion) and
    narrowphase (contact retention).\n
    Shape of ``(num_geoms,)`` and type :class:`float32`.
    """

    margin: wp.array | None = None
    """
    Surface offset [m] for each collision geometry.\n
    Pairwise additive.  Determines resting separation between shapes.\n
    Shape of ``(num_geoms,)`` and type :class:`float32`.
    """

    collidable_pairs: wp.array | None = None
    """
    Geometry-pair indices that are explicitly considered for collision detection.
    This array is used in broad-phase collision detection.\n
    Shape of ``(num_collidable_pairs,)`` and type :class:`vec2i`.
    """

    excluded_pairs: wp.array | None = None
    """
    Geometry-pair indices that are explicitly excluded from collision detection.\n
    This array is used in broad-phase collision detection.\n
    Shape of ``(num_excluded_geom_pairs,)`` and type :class:`vec2i`.
    """

    ###
    # Mesh / Heightfield Data (pass-through from Newton Model)
    ###

    heightfield_index: wp.array | None = None
    """Per-shape heightfield index (``-1`` for non-heightfield shapes)."""

    heightfield_data: wp.array | None = None
    """Concatenated :class:`HeightfieldData` structs for all heightfields."""

    heightfield_elevations: wp.array | None = None
    """Concatenated elevation samples for all heightfields."""

    collision_aabb_lower: wp.array | None = None
    """Per-shape local-space collision AABB lower bounds."""

    collision_aabb_upper: wp.array | None = None
    """Per-shape local-space collision AABB upper bounds."""

    voxel_resolution: wp.array | None = None
    """Per-shape voxel resolution for mesh contact reduction."""

    collision_radius: wp.array | None = None
    """Per-shape bounding-sphere radius for broadphase AABB computation."""


@dataclass
class GeometriesData:
    """
    An SoA-based container to hold time-varying data of a set of generic geometry entities.

    Attributes:
        num_geoms (int32): The total number of geometry entities in the model (host-side).
        pose (wp.array | None): The poses of the geometry entities in world coordinates.\n
            Shape of ``(num_geoms,)`` and type :class:`transformf`.
    """

    num_geoms: int = 0
    """Total number of geometry entities in the model (host-side)."""

    pose: wp.array | None = None
    """
    The poses of the geometry entities in world coordinates.\n
    Shape of ``(num_geoms,)`` and type :class:`transformf`.
    """


###
# Kernels
###


@wp.kernel
def _update_geometries_state(
    # Inputs:
    geom_bid: wp.array(dtype=int32),
    geom_offset: wp.array(dtype=transformf),
    body_pose: wp.array(dtype=transformf),
    # Outputs:
    geom_pose: wp.array(dtype=transformf),
):
    """
    A kernel to update poses of geometry entities in world
    coordinates from the poses of their associated bodies.

    **Inputs**:
        body_pose (wp.array):
            Array of per-body poses in world coordinates.\n
            Shape of ``(num_bodies,)`` and type :class:`transformf`.
        geom_bid (wp.array):
            Array of per-geom body indices.\n
            Shape of ``(num_geoms,)`` and type :class:`int32`.
        geom_offset (wp.array):
            Array of per-geom pose offsets w.r.t. their associated bodies.\n
            Shape of ``(num_geoms,)`` and type :class:`transformf`.

    **Outputs**:
        geom_pose (wp.array):
            Array of per-geom poses in world coordinates.\n
            Shape of ``(num_geoms,)`` and type :class:`transformf`.
    """
    # Retrieve the geometry index from the thread grid
    gid = wp.tid()

    # Retrieve the body index associated with the geometry
    bid = geom_bid[gid]

    # Retrieve the pose of the corresponding body
    X_b = wp.transform_identity(dtype=float32)
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
    body_poses: wp.array,
    geom_model: GeometriesModel,
    geom_data: GeometriesData,
):
    """
    Launches a kernel to update poses of geometry entities in
    world coordinates from the poses of their associated bodies.

    Args:
        body_poses (wp.array):
            The poses of the bodies in world coordinates.\n
            Shape of ``(num_bodies,)`` and type :class:`transformf`.
        geom_model (GeometriesModel):
            The model container holding time-invariant geometry data.
        geom_data (GeometriesData):
            The data container of the geometry elements.
    """
    wp.launch(
        _update_geometries_state,
        dim=geom_model.num_geoms,
        inputs=[geom_model.bid, geom_model.offset, body_poses],
        outputs=[geom_data.pose],
        device=body_poses.device,
    )
