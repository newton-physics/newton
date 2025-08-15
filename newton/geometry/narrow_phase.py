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

import warp as wp

from .primitive_collisions import (
    ContactPoint,
    get_box_box,
    get_capsule_box,
    get_capsule_capsule,
    get_plane_box,
    get_plane_capsule,
    get_plane_cylinder,
    get_plane_sphere,
    get_sphere_box,
    get_sphere_capsule,
    get_sphere_cylinder,
    get_sphere_sphere,
)
from .types import GEO_BOX, GEO_CAPSULE, GEO_CYLINDER, GEO_PLANE, GEO_SPHERE


@wp.struct
class WriteContactArgs:
    contact_pair: wp.array(dtype=wp.vec2i)
    contact_position: wp.array(dtype=wp.vec3)
    contact_normal: wp.array(
        dtype=wp.vec3
    )  # Pointing from pariId.x to pairId.y, represents z axis of local contact frame
    contact_penetration: wp.array(dtype=float)  # negative if bodies overlap
    contact_tangent: wp.array(dtype=wp.vec3)  # Represents x axis of local contact frame
    contact_count: wp.array(dtype=int)  # Number of active contacts after narrow
    cutoff: float
    pair: wp.vec2i


@wp.func
def contact_writer(contact: ContactPoint, args: WriteContactArgs):
    if contact.dist > args.cutoff:
        return

    cid = wp.atomic_add(args.contact_count, 0, 1)
    args.contact_pair[cid] = args.pair
    args.contact_position[cid] = contact.pos
    args.contact_normal[cid] = contact.normal
    args.contact_penetration[cid] = contact.dist
    args.contact_tangent[cid] = contact.tangent


@wp.kernel(enable_backward=False)
def narrow_phase(
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Maybe colliding pairs - usually provided by broad phase
    num_candidate_pair: wp.array(dtype=wp.int32, ndim=1),  # Size one array - usually provided by broad phase
    geom_types: wp.array(dtype=wp.int32, ndim=1),  # All geom types, pairs index into it
    geom_data: wp.array(dtype=wp.vec4, ndim=1),  # Geom data (radius etc.)
    geom_transform: wp.array(dtype=wp.transform, ndim=1),  # In world space
    geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
    # Outputs
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_position: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(
        dtype=wp.vec3
    ),  # Pointing from pariId.x to pairId.y, represents z axis of local contact frame
    contact_penetration: wp.array(dtype=float),  # negative if bodies overlap
    contact_tangent: wp.array(dtype=wp.vec3),  # Represents x axis of local contact frame
    contact_count: wp.array(dtype=int),  # Number of active contacts after narrow
):
    id = wp.tid()

    if id >= num_candidate_pair[0]:
        return

    pair = candidate_pair[id]
    geom_type_1 = geom_types[pair[0]]
    geom_type_2 = geom_types[pair[1]]

    if geom_type_1 > geom_type_2:
        tmp = geom_type_1
        geom_type_1 = geom_type_2
        geom_type_2 = tmp
        tmp = pair[0]
        pair[0] = pair[1]
        pair[1] = tmp

    data_1 = geom_data[pair[0]]
    data_2 = geom_data[pair[1]]
    transform_1 = geom_transform[pair[0]]
    transform_2 = geom_transform[pair[1]]

    cutoff_1 = geom_cutoff[pair[0]]
    cutoff_2 = geom_cutoff[pair[1]]
    cutoff = wp.max(cutoff_1, cutoff_2)

    writer_args = WriteContactArgs()
    writer_args.contact_pair = contact_pair
    writer_args.contact_position = contact_position
    writer_args.contact_normal = contact_normal
    writer_args.contact_penetration = contact_penetration
    writer_args.contact_tangent = contact_tangent
    writer_args.contact_count = contact_count
    writer_args.cutoff = cutoff
    writer_args.pair = pair

    # Plane-Sphere
    if geom_type_1 == GEO_SPHERE and geom_type_2 == GEO_PLANE:
        plane_normal = wp.transform_vector(transform_1, wp.vec3(0.0, 0.0, 1.0))
        plane_pos = wp.transform_get_translation(transform_1)
        sphere_center = wp.transform_get_translation(transform_2)
        sphere_radius = data_2.x
        wp.static(get_plane_sphere(contact_writer))(plane_normal, plane_pos, sphere_center, sphere_radius, writer_args)

    # Plane-Box
    elif geom_type_1 == GEO_BOX and geom_type_2 == GEO_PLANE:
        plane_normal = wp.transform_vector(transform_1, wp.vec3(0.0, 0.0, 1.0))
        plane_pos = wp.transform_get_translation(transform_1)
        box_center = wp.transform_get_translation(transform_2)
        box_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_2))
        box_half_sizes = wp.vec3(data_2.x, data_2.y, data_2.z)
        wp.static(get_plane_box(contact_writer))(
            plane_normal, plane_pos, box_center, box_rot, box_half_sizes, cutoff, writer_args
        )

    # Plane-Capsule
    elif geom_type_1 == GEO_CAPSULE and geom_type_2 == GEO_PLANE:
        plane_normal = wp.transform_vector(transform_1, wp.vec3(0.0, 0.0, 1.0))
        plane_pos = wp.transform_get_translation(transform_1)
        cap_center = wp.transform_get_translation(transform_2)
        cap_axis = wp.transform_vector(transform_2, wp.vec3(0.0, 0.0, 1.0))
        cap_radius = data_2.x
        cap_half_length = data_2.y
        wp.static(get_plane_capsule(contact_writer))(
            plane_normal, plane_pos, cap_center, cap_axis, cap_radius, cap_half_length, writer_args
        )

    # Plane-Cylinder
    elif geom_type_1 == GEO_CYLINDER and geom_type_2 == GEO_PLANE:
        plane_normal = wp.transform_vector(transform_1, wp.vec3(0.0, 0.0, 1.0))
        plane_pos = wp.transform_get_translation(transform_1)
        cylinder_center = wp.transform_get_translation(transform_2)
        cylinder_axis = wp.transform_vector(transform_2, wp.vec3(0.0, 0.0, 1.0))
        cylinder_radius = data_2.x
        cylinder_half_height = data_2.y
        wp.static(get_plane_cylinder(contact_writer))(
            plane_normal, plane_pos, cylinder_center, cylinder_axis, cylinder_radius, cylinder_half_height, writer_args
        )

    # Plane-Ellipsoid
    # elif geom_type_1 == GEO_ELLIPSOID and geom_type_2 == GEO_PLANE:
    #     plane_normal = wp.transform_vector(transform_1, wp.vec3(0.0, 0.0, 1.0))
    #     plane_pos = wp.transform_get_translation(transform_1)
    #     ellipsoid_center = wp.transform_get_translation(transform_2)
    #     ellipsoid_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_2))
    #     ellipsoid_radii = wp.vec3(data_2.x, data_2.y, data_2.z)
    #     wp.static(get_plane_ellipsoid(contact_writer))(
    #         plane_normal, plane_pos, ellipsoid_center, ellipsoid_rot, ellipsoid_radii, writer_args
    #     )

    # Sphere-Sphere
    elif geom_type_1 == GEO_SPHERE and geom_type_2 == GEO_SPHERE:
        sphere1_center = wp.transform_get_translation(transform_1)
        sphere1_radius = data_1.x
        sphere2_center = wp.transform_get_translation(transform_2)
        sphere2_radius = data_2.x
        wp.static(get_sphere_sphere(contact_writer))(
            sphere1_center, sphere1_radius, sphere2_center, sphere2_radius, writer_args
        )

    # Sphere-Box
    elif geom_type_1 == GEO_SPHERE and geom_type_2 == GEO_BOX:
        sphere_center = wp.transform_get_translation(transform_1)
        sphere_radius = data_1.x
        box_center = wp.transform_get_translation(transform_2)
        box_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_2))
        box_half_sizes = wp.vec3(data_2.x, data_2.y, data_2.z)
        wp.static(get_sphere_box(contact_writer))(
            sphere_center, sphere_radius, box_center, box_rot, box_half_sizes, cutoff, writer_args
        )

    # Sphere-Capsule
    elif geom_type_1 == GEO_SPHERE and geom_type_2 == GEO_CAPSULE:
        sphere_center = wp.transform_get_translation(transform_1)
        sphere_radius = data_1.x
        sphere_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_1))
        cap_center = wp.transform_get_translation(transform_2)
        cap_axis = wp.transform_vector(transform_2, wp.vec3(0.0, 0.0, 1.0))
        cap_radius = data_2.x
        cap_half_length = data_2.y
        cap_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_2))
        wp.static(get_sphere_capsule(contact_writer))(
            sphere_center,
            sphere_radius,
            sphere_rot,
            cap_center,
            cap_axis,
            cap_radius,
            cap_half_length,
            cap_rot,
            writer_args,
        )

    # Sphere-Cylinder
    elif geom_type_1 == GEO_SPHERE and geom_type_2 == GEO_CYLINDER:
        sphere_center = wp.transform_get_translation(transform_1)
        sphere_radius = data_1.x
        sphere_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_1))
        cylinder_center = wp.transform_get_translation(transform_2)
        cylinder_axis = wp.transform_vector(transform_2, wp.vec3(0.0, 0.0, 1.0))
        cylinder_radius = data_2.x
        cylinder_half_height = data_2.y
        cylinder_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_2))
        wp.static(get_sphere_cylinder(contact_writer))(
            sphere_center,
            sphere_radius,
            sphere_rot,
            cylinder_center,
            cylinder_axis,
            cylinder_radius,
            cylinder_half_height,
            cylinder_rot,
            writer_args,
        )

    # Box-Box
    elif geom_type_1 == GEO_BOX and geom_type_2 == GEO_BOX:
        box1_center = wp.transform_get_translation(transform_1)
        box1_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_1))
        box1_half_sizes = wp.vec3(data_1.x, data_1.y, data_1.z)
        box2_center = wp.transform_get_translation(transform_2)
        box2_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_2))
        box2_half_sizes = wp.vec3(data_2.x, data_2.y, data_2.z)
        wp.static(get_box_box(contact_writer))(
            box1_center, box1_rot, box1_half_sizes, box2_center, box2_rot, box2_half_sizes, cutoff, writer_args
        )

    # Capsule-Capsule
    elif geom_type_1 == GEO_CAPSULE and geom_type_2 == GEO_CAPSULE:
        cap1_center = wp.transform_get_translation(transform_1)
        cap1_axis = wp.transform_vector(transform_1, wp.vec3(0.0, 0.0, 1.0))
        cap1_radius = data_1.x
        cap1_half_length = data_1.y
        cap2_center = wp.transform_get_translation(transform_2)
        cap2_axis = wp.transform_vector(transform_2, wp.vec3(0.0, 0.0, 1.0))
        cap2_radius = data_2.x
        cap2_half_length = data_2.y
        wp.static(get_capsule_capsule(contact_writer))(
            cap1_center,
            cap1_axis,
            cap1_radius,
            cap1_half_length,
            cap2_center,
            cap2_axis,
            cap2_radius,
            cap2_half_length,
            writer_args,
        )

    # Capsule-Box
    elif geom_type_1 == GEO_BOX and geom_type_2 == GEO_CAPSULE:
        cap_center = wp.transform_get_translation(transform_1)
        cap_axis = wp.transform_vector(transform_1, wp.vec3(0.0, 0.0, 1.0))
        cap_radius = data_1.x
        cap_half_length = data_1.y
        box_center = wp.transform_get_translation(transform_2)
        box_rot = wp.quat_to_matrix(wp.transform_get_rotation(transform_2))
        box_half_sizes = wp.vec3(data_2.x, data_2.y, data_2.z)
        wp.static(get_capsule_box(contact_writer))(
            cap_center, cap_axis, cap_radius, cap_half_length, box_center, box_rot, box_half_sizes, cutoff, writer_args
        )


class NarrowPhaseContactGeneration:
    """Narrow phase collision detection that performs detailed contact generation between primitive shapes.

    This class takes candidate collision pairs (typically from a broad phase) and performs detailed
    collision detection between various primitive geometric shapes to generate contact information.
    It supports collision detection between planes, spheres, boxes, capsules, and cylinders.

    The narrow phase collision detection computes:
    - Contact positions in world space
    - Contact normals (pointing from first to second geometry in pair)
    - Contact penetration distances (negative if overlapping)
    - Contact tangent vectors for friction calculations

    Supported primitive collision pairs:
    - Plane vs Sphere, Box, Capsule, Cylinder
    - Sphere vs Sphere, Box, Capsule
    - Box vs Box, Capsule
    - Capsule vs Capsule

    The collision detection is performed using analytical methods for each primitive pair type.
    Multiple contacts may be generated for a single pair depending on the geometry types and
    configuration. The contacts are returned in world space coordinates.
    """

    def __init__(self):
        pass

    def launch(
        self,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Maybe colliding pairs - usually provided by broad phase
        num_candidate_pair: wp.array(dtype=wp.int32, ndim=1),  # Size one array - usually provided by broad phase
        geom_types: wp.array(dtype=wp.int32, ndim=1),  # All geom types, pairs index into it
        geom_data: wp.array(dtype=wp.vec4, ndim=1),  # Geom data (radius etc.)
        geom_transform: wp.array(dtype=wp.transform, ndim=1),  # In world space
        geom_source: wp.array(
            dtype=wp.uint64, ndim=1
        ),  # The index into the source array of meshes, sdfs etc, type define by geom_types
        geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
        # Outputs
        contact_pair: wp.array(dtype=wp.vec2i),
        contact_position: wp.array(dtype=wp.vec3),
        contact_normal: wp.array(
            dtype=wp.vec3
        ),  # Pointing from pariId.x to pairId.y, represents z axis of local contact frame
        contact_penetration: wp.array(dtype=float),  # negative if bodies overlap
        contact_tangent: wp.array(dtype=wp.vec3),  # Represents x axis of local contact frame
        contact_count: wp.array(dtype=int),  # Number of active contacts after narrow
    ):
        """Launch narrow phase collision detection to generate detailed contact information.

        This method processes candidate collision pairs and performs detailed collision detection
        between primitive geometric shapes. It outputs contact points with positions, normals,
        penetration distances, and tangent vectors for subsequent constraint solving.

        Args:
            candidate_pair: Array of geometry index pairs to check for collision, typically from broad phase
            num_candidate_pair: Single-element array containing the number of valid candidate pairs
            geom_types: Array of geometry type constants (GEOM_TYPE_SPHERE, GEOM_TYPE_BOX, etc.) for each geometry
            geom_data: Array of geometry-specific data (radius, half-extents, etc.) packed into vec4
            geom_transform: Array of world-space transforms for each geometry
            geom_source: Array of indices into source arrays for complex geometries (meshes, SDFs) - currently unused
            geom_cutoff: Array of collision cutoff distances per geometry
            contact_pair: Output array to store geometry pairs that generated contacts
            contact_position: Output array to store contact positions in world space
            contact_normal: Output array to store contact normals pointing from first to second geometry
            contact_penetration: Output array to store penetration distances (negative for overlap)
            contact_tangent: Output array to store contact tangent vectors for friction
            contact_count: Output array to store the total number of contacts generated

        The method will populate the output arrays with contact information for all overlapping primitive
        pairs within their cutoff distances. The contact_count[0] will contain the total number of
        contacts generated.
        """
        contact_count.zero_()
        wp.launch(
            narrow_phase,
            dim=candidate_pair.shape[0],
            inputs=[
                candidate_pair,
                num_candidate_pair,
                geom_types,
                geom_data,
                geom_transform,
                geom_cutoff,
                contact_pair,
                contact_position,
                contact_normal,
                contact_penetration,
                contact_tangent,
                contact_count,
            ],
        )
        # TODO: Add GJK narrow phase
        # wp.launch(gjk)

        # TODO: Add support for meshes and possible SDFs
