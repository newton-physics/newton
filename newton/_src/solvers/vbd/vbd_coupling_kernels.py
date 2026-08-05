# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels used by VBD's coupled-solver hooks."""

from __future__ import annotations

import warp as wp

from ...math import quat_velocity
from .particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    evaluate_edge_edge_contact_2_vertices,
    evaluate_vertex_triangle_collision_force_hessian_4_vertices,
)
from .rigid_vbd_kernels import _eval_body_particle_contact
from .tri_mesh_collision import TriMeshCollisionInfo

wp.set_module_options({"enable_backward": False})


@wp.kernel(enable_backward=False)
def _update_vbd_body_input_state_kernel(
    dt: float,
    body_flags: wp.array[wp.int32],
    kinematic_flag: int,
    body_world: wp.array[wp.int32],
    pose_rebaseline_mask: wp.array[wp.bool],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    local_body = wp.tid()

    world = body_world[local_body]
    if world < 0:
        world = pose_rebaseline_mask.shape[0] - 1
    if pose_rebaseline_mask[world]:
        # Accept first/reset poses before teleport conversion or the kinematic early exit.
        body_q_prev[local_body] = body_q[local_body]
        return

    if (body_flags[local_body] & kinematic_flag) != 0:
        return

    q_prev = body_q_prev[local_body]
    q_teleported = body_q[local_body]

    p_teleported = wp.transform_get_translation(q_teleported)
    p_prev = wp.transform_get_translation(q_prev)
    dv = (p_teleported - p_prev) / dt

    r_teleported = wp.transform_get_rotation(q_teleported)
    r_prev = wp.transform_get_rotation(q_prev)
    dw = quat_velocity(r_teleported, r_prev, dt)

    body_qd[local_body] += wp.spatial_vector(dv, dw)
    body_q[local_body] = q_prev


@wp.kernel(enable_backward=False)
def _harvest_vbd_proxy_wrenches_kernel(
    rigid_contact_count: wp.array[int],
    contact_body0: wp.array[wp.int32],
    contact_body1: wp.array[wp.int32],
    contact_point0_world: wp.array[wp.vec3],
    contact_point1_world: wp.array[wp.vec3],
    contact_force_on_body1: wp.array[wp.vec3],
    dst_body_inv_mass: wp.array[float],
    dst_body_flags: wp.array[wp.int32],
    body_local_to_proxy_global: wp.array[int],
    proxy_flag: int,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    out_proxy_body_f: wp.array[wp.spatial_vector],
):
    """Accumulate dynamic body-vs-proxy contact wrenches on proxy bodies."""
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    body0 = contact_body0[contact_id]
    body1 = contact_body1[contact_id]
    if body0 < 0 or body1 < 0:
        return

    is_proxy0 = int(0)
    is_proxy1 = int(0)
    proxy_global0 = int(-1)
    proxy_global1 = int(-1)
    if body0 < dst_body_flags.shape[0] and (dst_body_flags[body0] & proxy_flag) != 0:
        proxy_global0 = body_local_to_proxy_global[body0]
        if proxy_global0 >= 0:
            is_proxy0 = 1
    if body1 < dst_body_flags.shape[0] and (dst_body_flags[body1] & proxy_flag) != 0:
        proxy_global1 = body_local_to_proxy_global[body1]
        if proxy_global1 >= 0:
            is_proxy1 = 1

    if (is_proxy0 + is_proxy1) != 1:
        return

    other_id = body1 if is_proxy0 == 1 else body0
    if other_id < 0 or other_id >= dst_body_inv_mass.shape[0]:
        return
    if dst_body_inv_mass[other_id] <= 0.0:
        return

    force_on_b1 = contact_force_on_body1[contact_id]
    if is_proxy1 == 1:
        proxy_local_id = body1
        proxy_global_id = proxy_global1
        contact_point = contact_point1_world[contact_id]
        force_on_proxy = force_on_b1
    else:
        proxy_local_id = body0
        proxy_global_id = proxy_global0
        contact_point = contact_point0_world[contact_id]
        force_on_proxy = -force_on_b1

    if proxy_global_id < 0 or proxy_global_id >= out_proxy_body_f.shape[0]:
        return

    com_world = wp.transform_point(body_q[proxy_local_id], body_com[proxy_local_id])
    torque = wp.cross(contact_point - com_world, force_on_proxy)
    wp.atomic_add(out_proxy_body_f, proxy_global_id, wp.spatial_vector(force_on_proxy, torque))


@wp.kernel(enable_backward=False)
def _harvest_vbd_body_particle_contact_forces_on_proxy_bodies_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    friction_epsilon: float,
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    body_particle_contact_count: wp.array[int],
    body_particle_contact_particle: wp.array[int],
    body_particle_contact_shape: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_body_vel: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    shape_body: wp.array[wp.int32],
    out_body_f: wp.array[wp.spatial_vector],
):
    contact_idx = wp.tid()
    if contact_idx >= body_particle_contact_count[0]:
        return

    shape_idx = body_particle_contact_shape[contact_idx]
    if shape_idx < 0 or shape_idx >= shape_body.shape[0]:
        return

    body_idx = shape_body[shape_idx]
    if body_idx < 0 or body_idx >= body_local_to_proxy_global.shape[0]:
        return

    proxy_global = body_local_to_proxy_global[body_idx]
    if proxy_global < 0 or proxy_global >= out_body_f.shape[0]:
        return

    particle_idx = body_particle_contact_particle[contact_idx]
    if particle_idx < 0 or particle_idx >= particle_q.shape[0]:
        return

    force_on_particle, _ = _eval_body_particle_contact(
        particle_idx,
        particle_q[particle_idx],
        particle_q_prev[particle_idx],
        contact_idx,
        body_particle_contact_penalty_k[contact_idx],
        body_particle_contact_material_kd[contact_idx],
        body_particle_contact_material_mu[contact_idx],
        friction_epsilon,
        particle_radius,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        body_particle_contact_shape,
        body_particle_contact_body_pos,
        body_particle_contact_body_vel,
        body_particle_contact_normal,
        shape_margin,
        dt,
    )

    force_on_body = -force_on_particle
    cp_world = wp.transform_point(body_q[body_idx], body_particle_contact_body_pos[contact_idx])
    com_world = wp.transform_point(body_q[body_idx], body_com[body_idx])
    torque_on_body = wp.cross(cp_world - com_world, force_on_body)
    wp.atomic_add(out_body_f, proxy_global, wp.spatial_vector(force_on_body, torque_on_body))


@wp.func
def _vbd_particle_is_mapped_proxy(
    particle_idx: int,
    particle_local_to_proxy_global: wp.array[int],
    particle_flags: wp.array[wp.int32],
    proxy_particle_flag: int,
):
    if particle_idx < 0 or particle_idx >= particle_local_to_proxy_global.shape[0]:
        return False
    if particle_local_to_proxy_global[particle_idx] < 0:
        return False
    if (particle_flags[particle_idx] & proxy_particle_flag) == 0:
        return False
    return True


@wp.func
def _vbd_particle_is_dynamic_nonproxy(
    particle_idx: int,
    particle_flags: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    active_particle_flag: int,
    proxy_particle_flag: int,
):
    if particle_idx < 0 or particle_idx >= particle_flags.shape[0] or particle_idx >= particle_inv_mass.shape[0]:
        return False
    if (particle_flags[particle_idx] & active_particle_flag) == 0:
        return False
    if (particle_flags[particle_idx] & proxy_particle_flag) != 0:
        return False
    if particle_inv_mass[particle_idx] <= 0.0:
        return False
    return True


@wp.func
def _vbd_body_is_dynamic_nonproxy(
    body_idx: int,
    body_flags: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    proxy_body_flag: int,
):
    if body_idx < 0 or body_idx >= body_flags.shape[0] or body_idx >= body_inv_mass.shape[0]:
        return False
    if (body_flags[body_idx] & proxy_body_flag) != 0:
        return False
    if body_inv_mass[body_idx] <= 0.0:
        return False
    return True


@wp.func
def _vbd_add_proxy_particle_force(
    particle_idx: int,
    force: wp.vec3,
    particle_local_to_proxy_global: wp.array[int],
    out_particle_f: wp.array[wp.vec3],
):
    proxy_global = particle_local_to_proxy_global[particle_idx]
    if proxy_global < 0 or proxy_global >= out_particle_f.shape[0]:
        return
    wp.atomic_add(out_particle_f, proxy_global, force)


@wp.kernel(enable_backward=False)
def _harvest_vbd_proxy_particle_body_contact_forces_kernel(
    dt: float,
    particle_local_to_proxy_global: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    active_particle_flag: int,
    proxy_particle_flag: int,
    friction_epsilon: float,
    particle_radius: wp.array[float],
    body_particle_contact_count: wp.array[int],
    body_particle_contact_particle: wp.array[int],
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    proxy_body_flag: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_particle_contact_shape: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_body_vel: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    out_particle_f: wp.array[wp.vec3],
):
    contact_idx = wp.tid()
    if contact_idx >= body_particle_contact_count[0]:
        return

    particle_idx = body_particle_contact_particle[contact_idx]
    if particle_idx < 0:  # edge/face record: no single particle id (harvested via the per-body path)
        return
    if not _vbd_particle_is_mapped_proxy(
        particle_idx, particle_local_to_proxy_global, particle_flags, proxy_particle_flag
    ):
        return

    shape_idx = body_particle_contact_shape[contact_idx]
    if shape_idx < 0 or shape_idx >= shape_body.shape[0]:
        return

    body_idx = shape_body[shape_idx]
    if not _vbd_body_is_dynamic_nonproxy(body_idx, body_flags, body_inv_mass, proxy_body_flag):
        return

    body_contact_force, _ = _eval_body_particle_contact(
        particle_idx,
        particle_q[particle_idx],
        particle_q_prev[particle_idx],
        contact_idx,
        body_particle_contact_penalty_k[contact_idx],
        body_particle_contact_material_kd[contact_idx],
        body_particle_contact_material_mu[contact_idx],
        friction_epsilon,
        particle_radius,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        body_particle_contact_shape,
        body_particle_contact_body_pos,
        body_particle_contact_body_vel,
        body_particle_contact_normal,
        shape_margin,
        dt,
    )
    _vbd_add_proxy_particle_force(particle_idx, body_contact_force, particle_local_to_proxy_global, out_particle_f)


@wp.kernel(enable_backward=False)
def _harvest_vbd_proxy_particle_self_contact_forces_kernel(
    dt: float,
    particle_local_to_proxy_global: wp.array[int],
    particle_q_prev: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    active_particle_flag: int,
    proxy_particle_flag: int,
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    out_particle_f: wp.array[wp.vec3],
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id
        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]
                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

                e1_proxy = _vbd_particle_is_mapped_proxy(
                    e1_v1, particle_local_to_proxy_global, particle_flags, proxy_particle_flag
                ) and _vbd_particle_is_mapped_proxy(
                    e1_v2, particle_local_to_proxy_global, particle_flags, proxy_particle_flag
                )
                e2_dynamic = _vbd_particle_is_dynamic_nonproxy(
                    e2_v1, particle_flags, particle_inv_mass, active_particle_flag, proxy_particle_flag
                ) and _vbd_particle_is_dynamic_nonproxy(
                    e2_v2, particle_flags, particle_inv_mass, active_particle_flag, proxy_particle_flag
                )

                if e1_proxy and e2_dynamic:
                    has_contact, collision_force_0, collision_force_1, _hessian_0, _hessian_1 = (
                        evaluate_edge_edge_contact_2_vertices(
                            e1_idx,
                            e2_idx,
                            particle_q,
                            particle_q_prev,
                            edge_indices,
                            collision_radius,
                            soft_contact_ke,
                            soft_contact_kd,
                            soft_contact_mu,
                            friction_epsilon,
                            dt,
                            edge_edge_parallel_epsilon,
                        )
                    )

                    if has_contact:
                        _vbd_add_proxy_particle_force(
                            e1_v1, collision_force_0, particle_local_to_proxy_global, out_particle_f
                        )
                        _vbd_add_proxy_particle_force(
                            e1_v2, collision_force_1, particle_local_to_proxy_global, out_particle_f
                        )
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id
        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                vertex_proxy = _vbd_particle_is_mapped_proxy(
                    particle_idx, particle_local_to_proxy_global, particle_flags, proxy_particle_flag
                )
                vertex_dynamic = _vbd_particle_is_dynamic_nonproxy(
                    particle_idx, particle_flags, particle_inv_mass, active_particle_flag, proxy_particle_flag
                )
                tri_proxy = (
                    _vbd_particle_is_mapped_proxy(
                        tri_a, particle_local_to_proxy_global, particle_flags, proxy_particle_flag
                    )
                    and _vbd_particle_is_mapped_proxy(
                        tri_b, particle_local_to_proxy_global, particle_flags, proxy_particle_flag
                    )
                    and _vbd_particle_is_mapped_proxy(
                        tri_c, particle_local_to_proxy_global, particle_flags, proxy_particle_flag
                    )
                )
                tri_dynamic = (
                    _vbd_particle_is_dynamic_nonproxy(
                        tri_a, particle_flags, particle_inv_mass, active_particle_flag, proxy_particle_flag
                    )
                    and _vbd_particle_is_dynamic_nonproxy(
                        tri_b, particle_flags, particle_inv_mass, active_particle_flag, proxy_particle_flag
                    )
                    and _vbd_particle_is_dynamic_nonproxy(
                        tri_c, particle_flags, particle_inv_mass, active_particle_flag, proxy_particle_flag
                    )
                )

                if (vertex_proxy and tri_dynamic) or (tri_proxy and vertex_dynamic):
                    (
                        has_contact,
                        collision_force_0,
                        collision_force_1,
                        collision_force_2,
                        collision_force_3,
                        _hessian_0,
                        _hessian_1,
                        _hessian_2,
                        _hessian_3,
                    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                        particle_idx,
                        tri_idx,
                        particle_q,
                        particle_q_prev,
                        tri_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        soft_contact_mu,
                        friction_epsilon,
                        dt,
                    )

                    if has_contact:
                        if vertex_proxy and tri_dynamic:
                            _vbd_add_proxy_particle_force(
                                particle_idx, collision_force_3, particle_local_to_proxy_global, out_particle_f
                            )
                        if tri_proxy and vertex_dynamic:
                            _vbd_add_proxy_particle_force(
                                tri_a, collision_force_0, particle_local_to_proxy_global, out_particle_f
                            )
                            _vbd_add_proxy_particle_force(
                                tri_b, collision_force_1, particle_local_to_proxy_global, out_particle_f
                            )
                            _vbd_add_proxy_particle_force(
                                tri_c, collision_force_2, particle_local_to_proxy_global, out_particle_f
                            )
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE
