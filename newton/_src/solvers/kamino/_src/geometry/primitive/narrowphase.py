# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a narrow-phase Collision Detection (CD) backend optimized for geometric primitives.

This narrow-phase CD back-end uses the primitive colliders of Newton to compute
discrete contacts, but conforms to the data layout and required by Kamino.
"""

from typing import Any

import warp as wp

from ......geometry.collision_primitive import (
    MAXVAL,
    collide_box_box,
    collide_capsule_box,
    collide_capsule_capsule,
    collide_plane_box,
    collide_plane_capsule,
    collide_plane_cylinder,
    collide_plane_ellipsoid,
    collide_plane_sphere,
    collide_sphere_box,
    collide_sphere_capsule,
    collide_sphere_cylinder,
    collide_sphere_sphere,
)
from ......geometry.types import GeoType
from ...core.data import DataKamino
from ...core.materials import make_get_material_pair_properties
from ...core.model import ModelKamino
from ...core.types import (
    float32,
    int32,
    mat33f,
    quatf,
    transformf,
    uint32,
    uint64,
    vec2f,
    vec2i,
    vec3f,
    vec4f,
)
from ...geometry.contacts import ContactsKaminoData, make_contact_frame_znorm
from ...geometry.keying import build_pair_key2
from .broadphase import CollisionCandidatesData

###
# Module interface
###

__all__ = [
    "primitive_narrowphase",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS: list[tuple[GeoType, GeoType]] = [
    (GeoType.BOX, GeoType.BOX),
    (GeoType.CAPSULE, GeoType.BOX),
    (GeoType.CAPSULE, GeoType.CAPSULE),
    (GeoType.PLANE, GeoType.BOX),
    (GeoType.PLANE, GeoType.CAPSULE),
    (GeoType.PLANE, GeoType.CYLINDER),
    (GeoType.PLANE, GeoType.ELLIPSOID),
    (GeoType.PLANE, GeoType.SPHERE),
    (GeoType.SPHERE, GeoType.BOX),
    (GeoType.SPHERE, GeoType.CAPSULE),
    (GeoType.SPHERE, GeoType.CYLINDER),
    (GeoType.SPHERE, GeoType.SPHERE),
]
"""
List of primitive shape combinations supported by the primitive narrow-phase collider.
"""


###
# Geometry helper Types
###


@wp.struct
class Box:
    gid: int32
    bid: int32
    pos: vec3f
    rot: mat33f
    size: vec3f


@wp.struct
class Sphere:
    gid: int32
    bid: int32
    pos: vec3f
    rot: mat33f
    radius: float32


@wp.struct
class Capsule:
    gid: int32
    bid: int32
    pos: vec3f
    rot: mat33f
    axis: vec3f
    radius: float32
    half_length: float32


@wp.struct
class Cylinder:
    gid: int32
    bid: int32
    pos: vec3f
    rot: mat33f
    axis: vec3f
    radius: float32
    half_height: float32


@wp.struct
class Plane:
    gid: int32
    bid: int32
    pos: vec3f
    rot: mat33f
    normal: vec3f
    distance: float32
    width: float32
    length: float32


@wp.struct
class Ellipsoid:
    gid: int32
    bid: int32
    pos: vec3f
    rot: mat33f
    size: vec3f


@wp.func
def make_box(pose: transformf, params: vec3f, gid: int32, bid: int32) -> Box:
    box = Box()
    box.gid = gid
    box.bid = bid
    box.pos = wp.transform_get_translation(pose)
    box.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    box.size = vec3f(params[0], params[1], params[2])
    return box


@wp.func
def make_sphere(pose: transformf, params: vec3f, gid: int32, bid: int32) -> Sphere:
    sphere = Sphere()
    sphere.gid = gid
    sphere.bid = bid
    sphere.pos = wp.transform_get_translation(pose)
    sphere.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    sphere.radius = params[0]
    return sphere


@wp.func
def make_capsule(pose: transformf, params: vec3f, gid: int32, bid: int32) -> Capsule:
    capsule = Capsule()
    capsule.gid = gid
    capsule.bid = bid
    capsule.pos = wp.transform_get_translation(pose)
    rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    capsule.rot = rot_mat
    # Capsule axis is along the local Z-axis
    capsule.axis = vec3f(rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 2])
    capsule.radius = params[0]
    capsule.half_length = params[1]
    return capsule


@wp.func
def make_cylinder(pose: transformf, params: vec3f, gid: int32, bid: int32) -> Cylinder:
    cylinder = Cylinder()
    cylinder.gid = gid
    cylinder.bid = bid
    cylinder.pos = wp.transform_get_translation(pose)
    rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    cylinder.rot = rot_mat
    # Cylinder axis is along the local Z-axis
    cylinder.axis = vec3f(rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 2])
    cylinder.radius = params[0]
    cylinder.half_height = params[1]
    return cylinder


@wp.func
def make_plane(pose: transformf, params: vec3f, gid: int32, bid: int32) -> Plane:
    plane = Plane()
    plane.gid = gid
    plane.bid = bid
    plane.pos = wp.transform_get_translation(pose)
    plane.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    # Plane normal is extracted from the rotation matrix (assuming the plane's local Z-axis is the normal)
    plane.normal = vec3f(plane.rot[0, 2], plane.rot[1, 2], plane.rot[2, 2])
    # Plane distance is extracted from the position along the normal direction
    plane.distance = -(plane.pos.x * plane.normal.x + plane.pos.y * plane.normal.y + plane.pos.z * plane.normal.z)
    # Plane dimensions (width and length) stored in params[0:2]
    plane.width = params[0]
    plane.length = params[1]
    return plane


@wp.func
def make_ellipsoid(pose: transformf, params: vec3f, gid: int32, bid: int32) -> Ellipsoid:
    ellipsoid = Ellipsoid()
    ellipsoid.gid = gid
    ellipsoid.bid = bid
    ellipsoid.pos = wp.transform_get_translation(pose)
    ellipsoid.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    # Ellipsoid size (radii) stored in params[0:3]
    ellipsoid.size = vec3f(params[0], params[1], params[2])
    return ellipsoid


###
# Common Functions
###


@wp.func
def add_single_contact(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    wid: int32,
    gid_1: int32,
    gid_2: int32,
    bid_1: int32,
    bid_2: int32,
    margin_plus_gap: float32,
    margin: float32,
    distance: float32,
    position: vec3f,
    normal: vec3f,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Skip if the contact distance exceeds the detection threshold
    if (distance - margin_plus_gap) > 0.0:
        return

    # Increment the active contact counters
    mcid = wp.atomic_add(contact_model_num, 0, 1)
    wcid = wp.atomic_add(contact_world_num, wid, 1)

    # If within max allocated contacts, write the new contact data
    if mcid < model_max_contacts and wcid < world_max_contacts:
        # Perform A/B geom and body assignment
        # NOTE: We want the normal to always point from A to B,
        # and hence body B to be the "effected" body in the contact
        # so we have to ensure that bid_B is always non-negative
        if bid_2 < 0:
            gid_AB = vec2i(gid_2, gid_1)
            bid_AB = vec2i(bid_2, bid_1)
            normal = -normal
        else:
            gid_AB = vec2i(gid_1, gid_2)
            bid_AB = vec2i(bid_1, bid_2)

        # Compute absolute penetration distance
        distance_abs = wp.abs(distance)

        # The colliders compute the contact point in the middle, and thus to get the
        # per-geom contact points we need to offset by the penetration depth along the normal
        position_A = position + 0.5 * distance_abs * normal
        position_B = position - 0.5 * distance_abs * normal

        # Store margin-shifted distance in gapfunc.w: negative means penetration
        # past the resting separation, zero means at rest, positive means within
        # the detection gap but not yet at rest.
        d = distance - margin
        gapfunc = vec4f(normal.x, normal.y, normal.z, d)
        q_frame = wp.quat_from_matrix(make_contact_frame_znorm(normal))
        material = vec2f(friction, restitution)
        key = build_pair_key2(uint32(gid_AB[0]), uint32(gid_AB[1]))

        # Store the active contact output data
        contact_wid[mcid] = wid
        contact_cid[mcid] = wcid
        contact_gid_AB[mcid] = gid_AB
        contact_bid_AB[mcid] = bid_AB
        contact_position_A[mcid] = position_A
        contact_position_B[mcid] = position_B
        contact_gapfunc[mcid] = gapfunc
        contact_frame[mcid] = q_frame
        contact_material[mcid] = material
        contact_key[mcid] = key

    # Otherwise roll-back the atomic add if we exceeded limits
    else:
        wp.atomic_sub(contact_model_num, 0, 1)
        wp.atomic_sub(contact_world_num, wid, 1)


def make_add_multiple_contacts(MAX_CONTACTS: int, SHARED_NORMAL: bool):
    # Define the function to add multiple contacts
    @wp.func
    def add_multiple_contacts(
        # Inputs:
        model_max_contacts: int32,
        world_max_contacts: int32,
        wid: int32,
        gid_1: int32,
        gid_2: int32,
        bid_1: int32,
        bid_2: int32,
        margin_plus_gap: float32,
        margin: float32,
        friction: float32,
        restitution: float32,
        distances: wp.types.vector(MAX_CONTACTS, wp.float32),
        positions: wp.types.matrix((MAX_CONTACTS, 3), wp.float32),
        normals: Any,
        # Outputs:
        contact_model_num: wp.array[int32],
        contact_world_num: wp.array[int32],
        contact_wid: wp.array[int32],
        contact_cid: wp.array[int32],
        contact_gid_AB: wp.array[vec2i],
        contact_bid_AB: wp.array[vec2i],
        contact_position_A: wp.array[vec3f],
        contact_position_B: wp.array[vec3f],
        contact_gapfunc: wp.array[vec4f],
        contact_frame: wp.array[quatf],
        contact_material: wp.array[vec2f],
        contact_key: wp.array[uint64],
    ):
        # Count valid contacts (those within the detection threshold)
        num_contacts = wp.int32(0)
        for k in range(MAX_CONTACTS):
            if distances[k] != wp.inf and distances[k] <= margin_plus_gap:
                num_contacts += 1

        # Skip operation if no contacts were detected
        if num_contacts == 0:
            return

        # Perform A/B geom and body assignment
        # NOTE: We want the normal to always point from A to B,
        # and hence body B to be the "effected" body in the contact
        # so we have to ensure that bid_B is always non-negative
        if bid_2 < 0:
            gid_AB = vec2i(gid_2, gid_1)
            bid_AB = vec2i(bid_2, bid_1)
        else:
            gid_AB = vec2i(gid_1, gid_2)
            bid_AB = vec2i(bid_1, bid_2)

        # Increment the active contact counter
        mcio = wp.atomic_add(contact_model_num, 0, num_contacts)
        wcio = wp.atomic_add(contact_world_num, wid, num_contacts)

        # Retrieve the maximum number of contacts that can be stored for this geom pair
        max_num_contacts = wp.min(wp.min(model_max_contacts - mcio, world_max_contacts - wcio), num_contacts)

        # Create the common material for this contact set
        material = vec2f(friction, restitution)
        key = build_pair_key2(uint32(gid_AB[0]), uint32(gid_AB[1]))

        # Define a separate active contact index
        # NOTE: This is different from k since some contacts
        # may be not meet the criteria for being active
        active_contact_idx = wp.int32(0)

        # Add generated contacts data to the output arrays
        for k in range(MAX_CONTACTS):
            # Break if we've reached the maximum number of contacts for this geom pair
            if active_contact_idx >= max_num_contacts:
                break

            # If contact is valid, store it
            if distances[k] != wp.inf and distances[k] <= margin_plus_gap:
                # Compute the global contact index
                mcid = mcio + active_contact_idx

                # Extract contact data based on whether we have shared or per-contact normals
                distance = distances[k]
                position = vec3f(positions[k, 0], positions[k, 1], positions[k, 2])
                if wp.static(SHARED_NORMAL):
                    normal = normals
                else:
                    normal = vec3f(normals[k, 0], normals[k, 1], normals[k, 2])
                distance_abs = wp.abs(distance)

                # Adjust normal direction based on body assignment
                if bid_2 < 0:
                    normal = -normal

                # This collider computes the contact point in the middle, and thus to get the
                # per-geom contact we need to offset the contact point by the penetration depth
                position_A = position + 0.5 * normal * distance_abs
                position_B = position - 0.5 * normal * distance_abs

                # Store margin-shifted distance in gapfunc.w
                d = distance - margin
                gapfunc = vec4f(normal.x, normal.y, normal.z, d)
                q_frame = wp.quat_from_matrix(make_contact_frame_znorm(normal))

                # Store contact data
                contact_wid[mcid] = wid
                contact_cid[mcid] = wcio + active_contact_idx
                contact_gid_AB[mcid] = gid_AB
                contact_bid_AB[mcid] = bid_AB
                contact_position_A[mcid] = position_A
                contact_position_B[mcid] = position_B
                contact_gapfunc[mcid] = gapfunc
                contact_frame[mcid] = q_frame
                contact_material[mcid] = material
                contact_key[mcid] = key

                # Increment active contact index
                active_contact_idx += 1

        # Roll-back the atomic add if we exceeded limits
        if active_contact_idx < num_contacts:
            wp.atomic_sub(contact_model_num, 0, num_contacts - active_contact_idx)
            wp.atomic_sub(contact_world_num, wid, num_contacts - active_contact_idx)

    # Return the generated function
    return add_multiple_contacts


###
# Primitive Colliders
###


@wp.func
def sphere_sphere(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    sphere1: Sphere,
    sphere2: Sphere,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Run the respective collider function to detect sphere-sphere contacts
    distance, position, normal = collide_sphere_sphere(sphere1.pos, sphere1.radius, sphere2.pos, sphere2.radius)

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        sphere1.gid,
        sphere2.gid,
        sphere1.bid,
        sphere2.bid,
        margin_plus_gap,
        margin,
        distance,
        position,
        normal,
        friction,
        restitution,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def sphere_cylinder(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    sphere1: Sphere,
    cylinder2: Cylinder,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distance, position, normal = collide_sphere_cylinder(
        sphere1.pos,
        sphere1.radius,
        cylinder2.pos,
        cylinder2.axis,
        cylinder2.radius,
        cylinder2.half_height,
    )

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        sphere1.gid,
        cylinder2.gid,
        sphere1.bid,
        cylinder2.bid,
        margin_plus_gap,
        margin,
        distance,
        position,
        normal,
        friction,
        restitution,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def sphere_cone():
    pass


@wp.func
def sphere_capsule(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    sphere1: Sphere,
    capsule2: Capsule,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distance, position, normal = collide_sphere_capsule(
        sphere1.pos,
        sphere1.radius,
        capsule2.pos,
        capsule2.axis,
        capsule2.radius,
        capsule2.half_length,
    )

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        sphere1.gid,
        capsule2.gid,
        sphere1.bid,
        capsule2.bid,
        margin_plus_gap,
        margin,
        distance,
        position,
        normal,
        friction,
        restitution,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def sphere_box(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    sphere1: Sphere,
    box2: Box,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distance, position, normal = collide_sphere_box(sphere1.pos, sphere1.radius, box2.pos, box2.rot, box2.size)

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        sphere1.gid,
        box2.gid,
        sphere1.bid,
        box2.bid,
        margin_plus_gap,
        margin,
        distance,
        position,
        normal,
        friction,
        restitution,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def sphere_ellipsoid():
    pass


@wp.func
def cylinder_cylinder():
    pass


@wp.func
def cylinder_cone():
    pass


@wp.func
def cylinder_capsule():
    pass


@wp.func
def cylinder_box():
    pass


@wp.func
def cylinder_ellipsoid():
    pass


@wp.func
def cone_cone():
    pass


@wp.func
def cone_capsule():
    pass


@wp.func
def cone_box():
    pass


@wp.func
def cone_ellipsoid():
    pass


@wp.func
def capsule_capsule(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    capsule1: Capsule,
    capsule2: Capsule,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distance, position, normal = collide_capsule_capsule(
        capsule1.pos,
        capsule1.axis,
        capsule1.radius,
        capsule1.half_length,
        capsule2.pos,
        capsule2.axis,
        capsule2.radius,
        capsule2.half_length,
    )

    # Add the active contact to the global contacts arrays
    for k in range(2):
        if distance[k] != MAXVAL:
            add_single_contact(
                model_max_contacts,
                world_max_contacts,
                wid,
                capsule1.gid,
                capsule2.gid,
                capsule1.bid,
                capsule2.bid,
                margin_plus_gap,
                margin,
                distance[k],
                position[k],
                normal,
                friction,
                restitution,
                contact_model_num,
                contact_world_num,
                contact_wid,
                contact_cid,
                contact_gid_AB,
                contact_bid_AB,
                contact_position_A,
                contact_position_B,
                contact_gapfunc,
                contact_frame,
                contact_material,
                contact_key,
            )


@wp.func
def capsule_box(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    capsule1: Capsule,
    box2: Box,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normals = collide_capsule_box(
        capsule1.pos,
        capsule1.axis,
        capsule1.radius,
        capsule1.half_length,
        box2.pos,
        box2.rot,
        box2.size,
    )

    # Add the active contacts to the global contacts arrays (up to 2 contacts with per-contact normals)
    wp.static(make_add_multiple_contacts(2, False))(
        model_max_contacts,
        world_max_contacts,
        wid,
        capsule1.gid,
        box2.gid,
        capsule1.bid,
        box2.bid,
        margin_plus_gap,
        margin,
        friction,
        restitution,
        distances,
        positions,
        normals,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def capsule_ellipsoid():
    pass


@wp.func
def box_box(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    box1: Box,
    box2: Box,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normals = collide_box_box(
        box1.pos, box1.rot, box1.size, box2.pos, box2.rot, box2.size, margin_plus_gap
    )

    # Add the active contacts to the global contacts arrays (up to 8 contacts with per-contact normals)
    wp.static(make_add_multiple_contacts(8, False))(
        model_max_contacts,
        world_max_contacts,
        wid,
        box1.gid,
        box2.gid,
        box1.bid,
        box2.bid,
        margin_plus_gap,
        margin,
        friction,
        restitution,
        distances,
        positions,
        normals,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def box_ellipsoid():
    pass


@wp.func
def ellipsoid_ellipsoid():
    pass


@wp.func
def plane_sphere(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    plane1: Plane,
    sphere2: Sphere,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    # Note: collide_plane_sphere returns (distance, position) without normal
    distance, position = collide_plane_sphere(plane1.normal, plane1.pos, sphere2.pos, sphere2.radius)

    # Use plane normal as contact normal
    normal = plane1.normal

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        plane1.gid,
        sphere2.gid,
        plane1.bid,
        sphere2.bid,
        margin_plus_gap,
        margin,
        distance,
        position,
        normal,
        friction,
        restitution,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def plane_box(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    plane1: Plane,
    box2: Box,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normal = collide_plane_box(
        plane1.normal, plane1.pos, box2.pos, box2.rot, box2.size, margin_plus_gap
    )

    # Add the active contacts to the global contacts arrays (up to 4 contacts with shared normal)
    wp.static(make_add_multiple_contacts(4, True))(
        model_max_contacts,
        world_max_contacts,
        wid,
        plane1.gid,
        box2.gid,
        plane1.bid,
        box2.bid,
        margin_plus_gap,
        margin,
        friction,
        restitution,
        distances,
        positions,
        normal,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def plane_ellipsoid(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    plane1: Plane,
    ellipsoid2: Ellipsoid,
    wid: int32,
    margin_plus_gap: float32,
    margin: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distance, position, normal = collide_plane_ellipsoid(
        plane1.normal, plane1.pos, ellipsoid2.pos, ellipsoid2.rot, ellipsoid2.size
    )

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        plane1.gid,
        ellipsoid2.gid,
        plane1.bid,
        ellipsoid2.bid,
        margin_plus_gap,
        margin,
        distance,
        position,
        normal,
        friction,
        restitution,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


@wp.func
def plane_capsule(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    plane1: Plane,
    capsule2: Capsule,
    wid: int32,
    threshold: float32,
    rest_offset: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    # Note: collide_plane_capsule returns a contact frame, not individual normals
    distances, positions, frame = collide_plane_capsule(
        plane1.normal, plane1.pos, capsule2.pos, capsule2.axis, capsule2.radius, capsule2.half_length
    )

    # Manually add contacts since plane_capsule returns a contact frame instead of normals
    # Count valid contacts
    num_contacts = int32(0)
    for k in range(2):
        if distances[k] != wp.inf and distances[k] <= threshold:
            num_contacts += 1

    # Skip operation if no contacts were detected
    if num_contacts == 0:
        return

    # Extract normal from the contact frame (first column)
    normal = vec3f(frame[0, 0], frame[1, 0], frame[2, 0])

    # Perform A/B geom and body assignment
    # NOTE: We want the normal to always point from A to B,
    # and hence body B to be the "effected" body in the contact
    # so we have to ensure that bid_B is always non-negative
    if capsule2.bid < 0:
        gid_AB = vec2i(capsule2.gid, plane1.gid)
        bid_AB = vec2i(capsule2.bid, plane1.bid)
        normal = -normal
    else:
        gid_AB = vec2i(plane1.gid, capsule2.gid)
        bid_AB = vec2i(plane1.bid, capsule2.bid)

    # Increment the active contact counter
    mcio = wp.atomic_add(contact_model_num, 0, num_contacts)
    wcio = wp.atomic_add(contact_world_num, wid, num_contacts)

    # Retrieve the maximum number of contacts that can be stored
    max_num_contacts = wp.min(wp.min(model_max_contacts - mcio, world_max_contacts - wcio), num_contacts)

    # Create the common properties shared by all contacts in the current set
    q_frame = wp.quat_from_matrix(make_contact_frame_znorm(normal))
    material = vec2f(friction, restitution)
    key = build_pair_key2(uint32(gid_AB[0]), uint32(gid_AB[1]))

    # Add generated contacts data to the output arrays
    active_contact_idx = int32(0)
    for k in range(2):
        # Break if we've reached the maximum number of contacts
        if active_contact_idx >= max_num_contacts:
            break

        # If contact is valid, store it
        if distances[k] != wp.inf and distances[k] <= threshold:
            # Compute the global contact index
            mcid = mcio + active_contact_idx

            # Get contact data
            distance = distances[k]
            position = vec3f(positions[k, 0], positions[k, 1], positions[k, 2])
            distance_abs = wp.abs(distance)

            # Offset contact point by penetration depth
            position_A = position + 0.5 * normal * distance_abs
            position_B = position - 0.5 * normal * distance_abs

            # Generate the gap-function and coordinate frame for this contact
            gapfunc = vec4f(normal.x, normal.y, normal.z, distance - rest_offset)

            # Store contact data
            contact_wid[mcid] = wid
            contact_cid[mcid] = wcio + active_contact_idx
            contact_gid_AB[mcid] = gid_AB
            contact_bid_AB[mcid] = bid_AB
            contact_position_A[mcid] = position_A
            contact_position_B[mcid] = position_B
            contact_gapfunc[mcid] = gapfunc
            contact_frame[mcid] = q_frame
            contact_material[mcid] = material
            contact_key[mcid] = key

            # Increment active contact index
            active_contact_idx += 1


@wp.func
def plane_cylinder(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    plane1: Plane,
    cylinder2: Cylinder,
    wid: int32,
    threshold: float32,
    rest_offset: float32,
    friction: float32,
    restitution: float32,
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normal = collide_plane_cylinder(
        plane1.normal, plane1.pos, cylinder2.pos, cylinder2.axis, cylinder2.radius, cylinder2.half_height
    )

    # Add the active contacts to the global contacts arrays (up to 4 contacts with shared normal)
    wp.static(make_add_multiple_contacts(4, True))(
        model_max_contacts,
        world_max_contacts,
        wid,
        plane1.gid,
        cylinder2.gid,
        plane1.bid,
        cylinder2.bid,
        threshold,
        rest_offset,
        friction,
        restitution,
        distances,
        positions,
        normal,
        contact_model_num,
        contact_world_num,
        contact_wid,
        contact_cid,
        contact_gid_AB,
        contact_bid_AB,
        contact_position_A,
        contact_position_B,
        contact_gapfunc,
        contact_frame,
        contact_material,
        contact_key,
    )


###
# Kernels
###


@wp.kernel
def _primitive_narrowphase(
    # Inputs
    default_gap: float32,
    geom_bid: wp.array[int32],
    geom_sid: wp.array[int32],
    geom_mid: wp.array[int32],
    geom_params: wp.array[vec3f],
    geom_gap: wp.array[float32],
    geom_margin: wp.array[float32],
    geom_pose: wp.array[transformf],
    candidate_model_num_pairs: wp.array[int32],
    candidate_wid: wp.array[int32],
    candidate_geom_pair: wp.array[vec2i],
    contact_model_max_num: wp.array[int32],
    contact_world_max_num: wp.array[int32],
    material_restitution: wp.array[float32],
    material_static_friction: wp.array[float32],
    material_dynamic_friction: wp.array[float32],
    material_pair_restitution: wp.array[float32],
    material_pair_static_friction: wp.array[float32],
    material_pair_dynamic_friction: wp.array[float32],
    # Outputs:
    contact_model_num: wp.array[int32],
    contact_world_num: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gid_AB: wp.array[vec2i],
    contact_bid_AB: wp.array[vec2i],
    contact_position_A: wp.array[vec3f],
    contact_position_B: wp.array[vec3f],
    contact_gapfunc: wp.array[vec4f],
    contact_frame: wp.array[quatf],
    contact_material: wp.array[vec2f],
    contact_key: wp.array[uint64],
):
    # Retrieve the geom-pair index (gpid) from the thread grid
    gpid = wp.tid()

    # Skip if the thread id is greater than the number of pairs
    if gpid >= candidate_model_num_pairs[0]:
        return

    # Retrieve the world index
    wid = candidate_wid[gpid]

    # Retrieve the maximum number of contacts allocated
    model_max_contacts = contact_model_max_num[0]
    world_max_contacts = contact_world_max_num[wid]

    # Retrieve the geometry indices
    geom_pair = candidate_geom_pair[gpid]
    gid1 = geom_pair[0]
    gid2 = geom_pair[1]

    bid1 = geom_bid[gid1]
    sid1 = geom_sid[gid1]
    mid1 = geom_mid[gid1]
    params1 = geom_params[gid1]
    gap1 = geom_gap[gid1]
    margin1 = geom_margin[gid1]
    pose1 = geom_pose[gid1]

    bid2 = geom_bid[gid2]
    sid2 = geom_sid[gid2]
    mid2 = geom_mid[gid2]
    params2 = geom_params[gid2]
    gap2 = geom_gap[gid2]
    margin2 = geom_margin[gid2]
    pose2 = geom_pose[gid2]

    # Pairwise additive rest offset (margin) determines resting separation
    margin_12 = margin1 + margin2

    # Effective detection threshold: margin + gap (contacts accepted when
    # surface_distance <= margin_12 + gap_12)
    contact_gap_12 = wp.max(default_gap, gap1) + wp.max(default_gap, gap2)
    threshold_12 = margin_12 + contact_gap_12

    # Retrieve the material properties for the geom pair
    restitution_12, _, mu_12 = wp.static(make_get_material_pair_properties())(
        mid1,
        mid2,
        material_restitution,
        material_static_friction,
        material_dynamic_friction,
        material_pair_restitution,
        material_pair_static_friction,
        material_pair_dynamic_friction,
    )

    # TODO(team): static loop unrolling to remove unnecessary branching
    if sid1 == GeoType.SPHERE and sid2 == GeoType.SPHERE:
        sphere_sphere(
            model_max_contacts,
            world_max_contacts,
            make_sphere(pose1, params1, gid1, bid1),
            make_sphere(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.SPHERE and sid2 == GeoType.CYLINDER:
        sphere_cylinder(
            model_max_contacts,
            world_max_contacts,
            make_sphere(pose1, params1, gid1, bid1),
            make_cylinder(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.SPHERE and sid2 == GeoType.CONE:
        sphere_cone()

    elif sid1 == GeoType.SPHERE and sid2 == GeoType.CAPSULE:
        sphere_capsule(
            model_max_contacts,
            world_max_contacts,
            make_sphere(pose1, params1, gid1, bid1),
            make_capsule(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.SPHERE and sid2 == GeoType.BOX:
        sphere_box(
            model_max_contacts,
            world_max_contacts,
            make_sphere(pose1, params1, gid1, bid1),
            make_box(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.SPHERE and sid2 == GeoType.ELLIPSOID:
        sphere_ellipsoid()

    elif sid1 == GeoType.CYLINDER and sid2 == GeoType.CYLINDER:
        cylinder_cylinder()

    elif sid1 == GeoType.CYLINDER and sid2 == GeoType.CONE:
        cylinder_cone()

    elif sid1 == GeoType.CYLINDER and sid2 == GeoType.CAPSULE:
        cylinder_capsule()

    elif sid1 == GeoType.CYLINDER and sid2 == GeoType.BOX:
        cylinder_box()

    elif sid1 == GeoType.CYLINDER and sid2 == GeoType.ELLIPSOID:
        cylinder_ellipsoid()

    elif sid1 == GeoType.CONE and sid2 == GeoType.CONE:
        cone_cone()

    elif sid1 == GeoType.CONE and sid2 == GeoType.CAPSULE:
        cone_capsule()

    elif sid1 == GeoType.CONE and sid2 == GeoType.BOX:
        cone_box()

    elif sid1 == GeoType.CONE and sid2 == GeoType.ELLIPSOID:
        cone_ellipsoid()

    elif sid1 == GeoType.CAPSULE and sid2 == GeoType.CAPSULE:
        capsule_capsule(
            model_max_contacts,
            world_max_contacts,
            make_capsule(pose1, params1, gid1, bid1),
            make_capsule(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.CAPSULE and sid2 == GeoType.BOX:
        capsule_box(
            model_max_contacts,
            world_max_contacts,
            make_capsule(pose1, params1, gid1, bid1),
            make_box(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.CAPSULE and sid2 == GeoType.ELLIPSOID:
        capsule_ellipsoid()

    elif sid1 == GeoType.BOX and sid2 == GeoType.BOX:
        box_box(
            model_max_contacts,
            world_max_contacts,
            make_box(pose1, params1, gid1, bid1),
            make_box(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.BOX and sid2 == GeoType.ELLIPSOID:
        box_ellipsoid()

    elif sid1 == GeoType.ELLIPSOID and sid2 == GeoType.ELLIPSOID:
        ellipsoid_ellipsoid()

    # Plane collisions (plane is always geometry 1, other shapes are geometry 2)
    elif sid1 == GeoType.PLANE and sid2 == GeoType.SPHERE:
        plane_sphere(
            model_max_contacts,
            world_max_contacts,
            make_plane(pose1, params1, gid1, bid1),
            make_sphere(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.PLANE and sid2 == GeoType.BOX:
        plane_box(
            model_max_contacts,
            world_max_contacts,
            make_plane(pose1, params1, gid1, bid1),
            make_box(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.PLANE and sid2 == GeoType.ELLIPSOID:
        plane_ellipsoid(
            model_max_contacts,
            world_max_contacts,
            make_plane(pose1, params1, gid1, bid1),
            make_ellipsoid(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.PLANE and sid2 == GeoType.CAPSULE:
        plane_capsule(
            model_max_contacts,
            world_max_contacts,
            make_plane(pose1, params1, gid1, bid1),
            make_capsule(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )

    elif sid1 == GeoType.PLANE and sid2 == GeoType.CYLINDER:
        plane_cylinder(
            model_max_contacts,
            world_max_contacts,
            make_plane(pose1, params1, gid1, bid1),
            make_cylinder(pose2, params2, gid2, bid2),
            wid,
            threshold_12,
            margin_12,
            mu_12,
            restitution_12,
            contact_model_num,
            contact_world_num,
            contact_wid,
            contact_cid,
            contact_gid_AB,
            contact_bid_AB,
            contact_position_A,
            contact_position_B,
            contact_gapfunc,
            contact_frame,
            contact_material,
            contact_key,
        )


###
# Kernel Launcher
###


def primitive_narrowphase(
    model: ModelKamino,
    data: DataKamino,
    candidates: CollisionCandidatesData,
    contacts: ContactsKaminoData,
    default_gap: float | None = None,
):
    """
    Launches the narrow-phase collision detection kernel optimized for primitive shapes.

    Args:
        model: The model containing the collision geometries.
        data: The data containing the current state of the geometries.
        candidates: The collision container holding collision pairs.
        contacts: The contacts container to store detected contacts.
        default_gap: Default detection gap [m] applied as a floor to per-geometry gaps.
            If None, ``0.0`` is used.
    """
    if default_gap is None:
        default_gap = 0.0
    if not isinstance(default_gap, float):
        raise TypeError("default_gap must be of type `float`")

    wp.launch(
        _primitive_narrowphase,
        dim=candidates.num_model_geom_pairs,
        inputs=[
            float32(default_gap),
            model.geoms.bid,
            model.geoms.type,
            model.geoms.material,
            model.geoms.params,
            model.geoms.gap,
            model.geoms.margin,
            data.geoms.pose,
            candidates.model_num_collisions,
            candidates.wid,
            candidates.geom_pair,
            contacts.model_max_contacts,
            contacts.world_max_contacts,
            model.materials.restitution,
            model.materials.static_friction,
            model.materials.dynamic_friction,
            model.material_pairs.restitution,
            model.material_pairs.static_friction,
            model.material_pairs.dynamic_friction,
        ],
        outputs=[
            contacts.model_active_contacts,
            contacts.world_active_contacts,
            contacts.wid,
            contacts.cid,
            contacts.gid_AB,
            contacts.bid_AB,
            contacts.position_A,
            contacts.position_B,
            contacts.gapfunc,
            contacts.frame,
            contacts.material,
            contacts.key,
        ],
        device=model.device,
    )
