# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Geometry Model Types & Containers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .....utils.heightfield import HeightfieldData

###
# Module interface
###

__all__ = [
    "GeometriesData",
    "GeometriesModel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Base Geometry Containers
###


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
    Collision group assigned to each collision geometry. These groups are based
    on Newton's collision group semantics.

    Group `0` will not collide with anything. Any positive group N will collide
    with the same group as well as any negative group. Any negative group -M
    will collide with all groups except -M. See docs/concepts/collisions.rst
    for details.

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
