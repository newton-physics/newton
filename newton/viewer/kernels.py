"""
Warp kernels for simplified Newton viewers.
These kernels handle mesh operations and transformations.
"""

import warp as wp


@wp.kernel
def compute_pick_state_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    hit_point_world: wp.vec3,
    # output
    pick_body: wp.array(dtype=int),
    pick_state: wp.array(dtype=float),
):
    if body_index < 0:
        return

    # store body index
    pick_body[0] = body_index

    # store target world
    pick_state[3] = hit_point_world[0]
    pick_state[4] = hit_point_world[1]
    pick_state[5] = hit_point_world[2]

    # compute and store local space attachment point
    X_wb = body_q[body_index]
    X_bw = wp.transform_inverse(X_wb)
    pick_pos_local = wp.transform_point(X_bw, hit_point_world)

    pick_state[0] = pick_pos_local[0]
    pick_state[1] = pick_pos_local[1]
    pick_state[2] = pick_pos_local[2]


@wp.kernel
def apply_picking_force_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    pick_body_arr: wp.array(dtype=int),
    pick_state: wp.array(dtype=float),
):
    pick_body = pick_body_arr[0]
    if pick_body < 0:
        return

    pick_pos_local = wp.vec3(pick_state[0], pick_state[1], pick_state[2])
    pick_target_world = wp.vec3(pick_state[3], pick_state[4], pick_state[5])
    pick_stiffness = pick_state[6]
    pick_damping = pick_state[7]
    angular_damping = 1.0  # Damping coefficient for angular velocity

    # world space attachment point
    X_wb = body_q[pick_body]
    pick_pos_world = wp.transform_point(X_wb, pick_pos_local)

    # center of mass
    com = wp.transform_get_translation(X_wb)

    # get velocity of attachment point
    omega = wp.spatial_top(body_qd[pick_body])
    vel_com = wp.spatial_bottom(body_qd[pick_body])
    vel_world = vel_com + wp.cross(omega, pick_pos_world - com)

    # compute spring force
    f = pick_stiffness * (pick_target_world - pick_pos_world) - pick_damping * vel_world

    # compute torque with angular damping
    t = wp.cross(pick_pos_world - com, f) - angular_damping * omega

    # apply force and torque
    wp.atomic_add(body_f, pick_body, wp.spatial_vector(t, f))


@wp.kernel
def update_pick_target_kernel(
    p: wp.vec3,
    d: wp.vec3,
    # read-write
    pick_state: wp.array(dtype=float),
):
    # get current target position
    current_target = wp.vec3(pick_state[3], pick_state[4], pick_state[5])

    # compute distance from ray origin to current target
    dist = wp.length(current_target - p)

    # project new target onto sphere with same radius
    new_target = p + d * dist

    pick_state[3] = new_target[0]
    pick_state[4] = new_target[1]
    pick_state[5] = new_target[2]


@wp.kernel
def compute_contact_points(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    # outputs
    contact_pos0: wp.array(dtype=wp.vec3),
    contact_pos1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        contact_pos0[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        contact_pos1[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        contact_pos0[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        contact_pos1[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        return

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    contact_pos0[tid] = wp.transform_point(X_wb_a, contact_point0[tid])
    contact_pos1[tid] = wp.transform_point(X_wb_b, contact_point1[tid])


@wp.kernel
def update_shape_xforms(
    shape_xforms: wp.array(dtype=wp.transform),
    shape_parents: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    world_xforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()

    shape_xform = shape_xforms[tid]
    shape_parent = shape_parents[tid]

    if shape_parent >= 0:
        world_xform = wp.transform_multiply(body_q[shape_parent], shape_xform)
    else:
        world_xform = shape_xform

    world_xforms[tid] = world_xform
