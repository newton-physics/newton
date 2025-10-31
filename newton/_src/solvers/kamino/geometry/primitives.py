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

"""Narrow-phase collision detection operations on geometric primitives"""

from __future__ import annotations

import warp as wp

from ....geometry.collision_primitive import (
    collide_box_box,
    collide_sphere_box,
    collide_sphere_sphere,
)
from ..core.model import Model, ModelData
from ..core.shapes import ShapeType
from ..core.types import (
    float32,
    int32,
    mat33f,
    transformf,
    vec2f,
    vec2i,
    vec3f,
    vec4f,
)
from ..geometry.collisions import Collisions
from ..geometry.contacts import Contacts
from ..geometry.math import make_contact_frame_znorm

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

DEFAULT_MARGIN = wp.constant(float32(1e-5))


###
# Geometry helper Types
###


@wp.struct
class Box:
    bid: int32
    gid: int32
    pos: vec3f
    rot: mat33f
    size: vec3f


@wp.struct
class Sphere:
    bid: int32
    gid: int32
    pos: vec3f
    rot: mat33f
    radius: float32


@wp.func
def make_box(pose: transformf, params: vec4f, gid: int32, bid: int32) -> Box:
    box = Box()
    box.bid = bid
    box.gid = gid
    box.pos = wp.transform_get_translation(pose)
    box.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    box.size = vec3f(0.5 * params[0], 0.5 * params[1], 0.5 * params[2])
    return box


@wp.func
def make_sphere(pose: transformf, params: vec4f, gid: int32, bid: int32) -> Sphere:
    sphere = Sphere()
    sphere.bid = bid
    sphere.gid = gid
    sphere.pos = wp.transform_get_translation(pose)
    sphere.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    sphere.radius = params[0]
    return sphere


###
# Utility Functions
###


@wp.func
def add_single_contact(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    wid_in: int32,
    bid_1_in: int32,
    bid_2_in: int32,
    margin_in: float32,
    distance_in: float32,
    position_in: vec3f,
    normal_in: vec3f,
    friction_in: float32,
    restitution_in: float32,
    # Outputs:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Skip if the contact distance exceeds the specified margin
    if (distance_in - margin_in) > 0.0:
        return

    # TODO: This will cause problems if the number of contacts exceeds the maximum as we are
    # incrementing the contact counters and do not decrement if we've exceeded the maximum
    mcid = wp.atomic_add(contacts_model_num_out, 0, 1)
    wcid = wp.atomic_add(contacts_world_num_out, wid_in, 1)
    if mcid < model_max_contacts_in and wcid < world_max_contacts_in:
        # Perform body assignment
        if bid_2_in < 0:
            bid_A_in = bid_2_in
            bid_B_in = bid_1_in
            normal_in = -normal_in
        else:
            bid_A_in = bid_1_in
            bid_B_in = bid_2_in

        # Compute absolute penetration distance
        distance_abs = wp.abs(distance_in)

        # The colliders compute the contact point in the middle, and thus to get the
        # per-geom contact points we need to offset by the penetration depth along the normal
        position_A_in = position_in + 0.5 * distance_abs * normal_in
        position_B_in = position_in - 0.5 * distance_abs * normal_in

        # Store the active contact output data
        contact_wid_out[mcid] = wid_in
        contact_cid_out[mcid] = wcid
        contact_body_A_out[mcid] = vec4f(position_A_in[0], position_A_in[1], position_A_in[2], float32(bid_A_in))
        contact_body_B_out[mcid] = vec4f(position_B_in[0], position_B_in[1], position_B_in[2], float32(bid_B_in))
        contact_gapfunc_out[mcid] = vec4f(normal_in[0], normal_in[1], normal_in[2], distance_in)
        contact_frame_out[mcid] = make_contact_frame_znorm(normal_in)
        contact_material_out[mcid][0] = friction_in
        contact_material_out[mcid][1] = restitution_in


def make_add_multiple_contacts(MAX_CONTACTS: int):
    # Define the function to add multiple contacts
    @wp.func
    def add_multiple_contacts(
        # Inputs:
        model_max_contacts_in: int32,
        world_max_contacts_in: int32,
        wid_in: int32,
        bid_1_in: int32,
        bid_2_in: int32,
        margin_in: float32,
        distances_in: wp.types.vector(MAX_CONTACTS, wp.float32),
        positions_in: wp.types.matrix((MAX_CONTACTS, 3), wp.float32),
        normals_in: wp.types.matrix((MAX_CONTACTS, 3), wp.float32),
        friction_in: float32,
        restitution_in: float32,
        # Outputs:
        contacts_model_num_out: wp.array(dtype=int32),
        contacts_world_num_out: wp.array(dtype=int32),
        contact_wid_out: wp.array(dtype=int32),
        contact_cid_out: wp.array(dtype=int32),
        contact_body_A_out: wp.array(dtype=vec4f),
        contact_body_B_out: wp.array(dtype=vec4f),
        contact_gapfunc_out: wp.array(dtype=vec4f),
        contact_frame_out: wp.array(dtype=mat33f),
        contact_material_out: wp.array(dtype=vec2f),
    ):
        # Count valid contacts (those with finite distance)
        num_contacts = wp.int32(0)
        for i in range(MAX_CONTACTS):
            if distances_in[i] != wp.inf and distances_in[i] <= margin_in:
                num_contacts += 1

        # Skip operation if no contacts were detected
        if num_contacts == 0:
            return

        # Assign body indices
        if bid_2_in < 0:
            bid_A = bid_2_in
            bid_B = bid_1_in
        else:
            bid_A = bid_1_in
            bid_B = bid_2_in

        # Increment the active contact counter
        mcio = wp.atomic_add(contacts_model_num_out, 0, num_contacts)
        wcio = wp.atomic_add(contacts_world_num_out, wid_in, num_contacts)

        # Retrieve the maximum number of contacts that can be stored for this geom pair
        max_num_contacts = wp.min(wp.min(model_max_contacts_in - mcio, world_max_contacts_in - wcio), num_contacts)

        # Add generated contacts data to the output arrays
        contact_idx = wp.int32(0)
        for i in range(8):
            # Break if we've reached the maximum number of contacts for this geom pair
            if contact_idx >= max_num_contacts:
                break

            # If contact is valid, store it
            if distances_in[i] != wp.inf and distances_in[i] <= margin_in:
                # Compute the global contact index
                mcid = mcio + contact_idx

                # Get contact data
                dist = distances_in[i]
                pos = vec3f(positions_in[i, 0], positions_in[i, 1], positions_in[i, 2])
                normal = vec3f(normals_in[i, 0], normals_in[i, 1], normals_in[i, 2])
                dist_abs = wp.abs(dist)

                # Adjust normal direction based on body assignment
                if bid_2_in < 0:
                    normal = -normal

                # Generate a contact frame
                frame = make_contact_frame_znorm(normal)

                # This collider computes the contact point in the middle, and thus to get the
                # per-geom contact we need to offset the contact point by the penetration depth
                pos_A = pos + 0.5 * normal * dist_abs
                pos_B = pos - 0.5 * normal * dist_abs

                # Store contact data
                contact_wid_out[mcid] = wid_in
                contact_cid_out[mcid] = wcio + contact_idx
                contact_body_A_out[mcid] = vec4f(pos_A[0], pos_A[1], pos_A[2], float32(bid_A))
                contact_body_B_out[mcid] = vec4f(pos_B[0], pos_B[1], pos_B[2], float32(bid_B))
                contact_gapfunc_out[mcid] = vec4f(normal[0], normal[1], normal[2], dist)
                contact_frame_out[mcid] = frame
                contact_material_out[mcid] = vec2f(friction_in, restitution_in)

                # Increment active contact index
                contact_idx += 1

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
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Run the respective collider function to detect sphere-sphere contacts
    distance, position, normal = collide_sphere_sphere(sphere1.pos, sphere1.radius, sphere2.pos, sphere2.radius)

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        sphere1.bid,
        sphere2.bid,
        margin,
        distance,
        position,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def sphere_cylinder():
    pass


@wp.func
def sphere_cone():
    pass


@wp.func
def sphere_capsule():
    pass


@wp.func
def sphere_box(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    sphere_in: Sphere,
    box_in: Box,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    dist, pos, normal = collide_sphere_box(sphere_in.pos, sphere_in.radius, box_in.pos, box_in.rot, box_in.size)

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        sphere_in.bid,
        box_in.bid,
        margin,
        dist,
        pos,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
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
def capsule_capsule():
    pass


@wp.func
def capsule_box():
    pass


@wp.func
def capsule_ellipsoid():
    pass


@wp.func
def box_box(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    box1_in: Box,
    box2_in: Box,
    wid_in: int32,
    margin_in: float32,
    friction_in: float32,
    restitution_in: float32,
    # Outputs:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normals = collide_box_box(
        box1_in.pos, box1_in.rot, box1_in.size, box2_in.pos, box2_in.rot, box2_in.size, margin_in
    )

    # Add the active contacts to the global contacts arrays
    wp.static(make_add_multiple_contacts(8))(
        # Inputs:
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        box1_in.bid,
        box2_in.bid,
        margin_in,
        distances,
        positions,
        normals,
        friction_in,
        restitution_in,
        # Outputs:
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def box_ellipsoid():
    pass


@wp.func
def ellipsoid_ellipsoid():
    pass


###
# Kernels
###


@wp.kernel
def _primitive_narrowphase(
    # Inputs
    geom_bid_in: wp.array(dtype=int32),
    geom_sid_in: wp.array(dtype=int32),
    geom_params_in: wp.array(dtype=vec4f),
    geom_offset_in: wp.array(dtype=transformf),
    geom_mid_in: wp.array(dtype=int32),
    geom_pose_in: wp.array(dtype=transformf),
    col_model_num_pairs_in: wp.array(dtype=int32),
    col_world_num_pairs_in: wp.array(dtype=int32),
    col_wid_in: wp.array(dtype=int32),
    col_geom_pair_in: wp.array(dtype=vec2i),
    contacts_model_max_num_in: wp.array(dtype=int32),
    contacts_world_max_num_in: wp.array(dtype=int32),
    # Outputs:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contacts_wid_out: wp.array(dtype=int32),
    contacts_cid_out: wp.array(dtype=int32),
    contacts_body_A_out: wp.array(dtype=vec4f),
    contacts_body_B_out: wp.array(dtype=vec4f),
    contacts_gapfunc_out: wp.array(dtype=vec4f),
    contacts_frame_out: wp.array(dtype=mat33f),
    contacts_material_out: wp.array(dtype=vec2f),
):
    # Retrieve the thread id
    tid = wp.tid()

    # Skip if the thread id is greater than the number of pairs
    if tid >= col_model_num_pairs_in[0]:
        return

    # Retrieve the world index
    wid = col_wid_in[tid]

    # Retrieve the maximum number of contacts for this world
    model_max_contacts = contacts_model_max_num_in[0]
    world_max_contacts = contacts_world_max_num_in[wid]

    # Retrieve the geometry ids
    geom_pair = col_geom_pair_in[tid]
    gid1 = geom_pair[0]
    gid2 = geom_pair[1]

    # Retrieve the body ids
    bid1 = geom_bid_in[gid1]
    bid2 = geom_bid_in[gid2]

    # Retrieve the shape ids
    sid1 = geom_sid_in[gid1]
    sid2 = geom_sid_in[gid2]

    # TODO(team): static loop unrolling to remove unnecessary branching
    if sid1 == ShapeType.SPHERE and sid2 == ShapeType.SPHERE:
        sphere_sphere(
            model_max_contacts,
            world_max_contacts,
            make_sphere(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_sphere(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.CYLINDER.value):
        sphere_cylinder()

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.CONE.value):
        sphere_cone()

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.CAPSULE.value):
        sphere_capsule()

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.BOX.value):
        sphere_box(
            model_max_contacts,
            world_max_contacts,
            make_sphere(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_box(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        sphere_ellipsoid()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.CYLINDER.value):
        cylinder_cylinder()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.CONE.value):
        cylinder_cone()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.CAPSULE.value):
        cylinder_capsule()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.BOX.value):
        cylinder_box()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        cylinder_ellipsoid()

    elif sid1 == int32(ShapeType.CONE.value) and sid2 == int32(ShapeType.CONE.value):
        cone_cone()

    elif sid1 == int32(ShapeType.CONE.value) and sid2 == int32(ShapeType.CAPSULE.value):
        cone_capsule()

    elif sid1 == int32(ShapeType.CONE.value) and sid2 == int32(ShapeType.BOX.value):
        cone_box()

    elif sid1 == int32(ShapeType.CONE.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        cone_ellipsoid()

    elif sid1 == int32(ShapeType.CAPSULE.value) and sid2 == int32(ShapeType.CAPSULE.value):
        capsule_capsule()

    elif sid1 == int32(ShapeType.CAPSULE.value) and sid2 == int32(ShapeType.BOX.value):
        capsule_box()

    elif sid1 == int32(ShapeType.CAPSULE.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        capsule_ellipsoid()

    elif sid1 == int32(ShapeType.BOX.value) and sid2 == int32(ShapeType.BOX.value):
        box_box(
            model_max_contacts,
            world_max_contacts,
            make_box(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_box(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )
    elif sid1 == int32(ShapeType.BOX.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        box_ellipsoid()
    elif sid1 == int32(ShapeType.ELLIPSOID.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        ellipsoid_ellipsoid()


###
# Kernel Launcher
###


def primitive_narrowphase(model: Model, state: ModelData, collisions: Collisions, contacts: Contacts):
    """
    Launches the narrow-phase collision detection kernel for primitive shapes.

    Arguments
    ------
        model (Model): The model containing the collision geometries.
        state (ModelData): The current state of the model.
        collisions (Collisions): The collision container holding collision pairs.
        contacts (Contacts): The contacts container to store detected contacts.
    """
    wp.launch(
        _primitive_narrowphase,
        dim=collisions.cmodel.num_model_geom_pairs,
        inputs=[
            # Inputs:
            model.cgeoms.bid,
            model.cgeoms.sid,
            model.cgeoms.params,
            model.cgeoms.offset,
            model.cgeoms.mid,
            state.cgeoms.pose,
            collisions.cdata.model_num_collisions,
            collisions.cdata.world_num_collisions,
            collisions.cdata.wid,
            collisions.cdata.geom_pair,
            contacts.model_max_contacts,
            contacts.world_max_contacts,
            # Outputs:
            contacts.model_num_contacts,
            contacts.world_num_contacts,
            contacts.wid,
            contacts.cid,
            contacts.body_A,
            contacts.body_B,
            contacts.gapfunc,
            contacts.frame,
            contacts.material,
        ],
    )
