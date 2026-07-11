# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SDF local-optimization primitives for full-surface rigid-soft contact.

Implements the Macklin et al. (2020) "Local Optimization for Robust Signed Distance Field
Collision" face (barycentric simplex) and edge (1-D golden-section) optimizers over a rigid
shape's signed-distance field, plus the shape-agnostic ``eval_shape_sdf`` dispatch.

This lives in its own module (not ``kernels.py``) because it needs the volume-SDF sampler from
``sdf_texture``, and ``sdf_texture -> sdf_utils -> kernels`` would make a ``kernels`` import cyclic.
Nothing imports this module except the collision launcher, so it is cycle-free.
"""

import warp as wp

from .flags import ShapeFlags
from .kernels import (
    counter_increment,
    sdf_box,
    sdf_box_grad,
    sdf_capsule,
    sdf_capsule_grad,
    sdf_cone,
    sdf_cone_grad,
    sdf_cylinder,
    sdf_cylinder_grad,
    sdf_ellipsoid,
    sdf_ellipsoid_grad,
    sdf_plane,
    sdf_sphere,
    sdf_sphere_grad,
)
from .sdf_texture import TextureSDFData, texture_sample_sdf_grad
from .types import Axis, GeoType

# Fixed iteration counts -> data-independent loops -> CUDA-graph-capturable. Passed as kernel args
# (uniform across threads/launches). Tuned against a brute-force grid reference
# (newton/tests/test_collision_pipeline.py, TestFullSurfaceSoftContact): edge golden-section is
# accurate (~1e-4); the face
# Frank-Wolfe tail is ~O(1/iters) (~3e-3 at 24 iters), sufficient for contact within margin.
SDF_EDGE_ITERS = 24
SDF_FACE_ITERS = 24
SDF_LS_ITERS = 16


@wp.func
def _is_analytic(geo: wp.int32):
    """True for primitives that evaluate phi/grad in closed form (no volume SDF needed)."""
    return (
        geo == GeoType.SPHERE
        or geo == GeoType.BOX
        or geo == GeoType.CAPSULE
        or geo == GeoType.CYLINDER
        or geo == GeoType.CONE
        or geo == GeoType.ELLIPSOID
        or geo == GeoType.PLANE
    )


@wp.func
def eval_shape_sdf(
    geo: wp.int32,
    scale: wp.vec3,
    x_local: wp.vec3,
    shape_sdf_index: wp.int32,
    texture_sdf_table: wp.array[TextureSDFData],
):
    """Return ``(phi_lower, phi, grad)`` for the rigid shape at shape-local ``x_local``.

    ``phi`` is the accurate shape-local signed distance -- used for the contact accept and the
    ``x - phi * grad`` surface projection. ``phi_lower`` is a conservative *lower bound* on it, used
    only for the candidate cull and the optimizer's internal search, so an under-estimate can never
    drop a real contact (the reject happens in the narrow phase on ``phi``). The two coincide for
    analytic primitives (closed form, exact) and for a uniform scale; they differ only for a
    nonuniformly scaled volume SDF, where the local distance stretches along the surface normal
    (``phi``) while the smallest scale magnitude is a cheap lower bound (``phi_lower``). ``grad`` is
    the unit outward gradient. Analytic primitives evaluate closed-form (Z-up, matching
    ``create_soft_contacts``); shapes with a provisioned volume SDF (``shape_sdf_index >= 0``) sample
    the texture SDF with query-time scaling.
    """
    if geo == GeoType.SPHERE:
        p = sdf_sphere(x_local, scale[0])
        return p, p, sdf_sphere_grad(x_local, scale[0])
    if geo == GeoType.BOX:
        p = sdf_box(x_local, scale[0], scale[1], scale[2])
        return p, p, sdf_box_grad(x_local, scale[0], scale[1], scale[2])
    if geo == GeoType.CAPSULE:
        p = sdf_capsule(x_local, scale[0], scale[1], int(Axis.Z))
        return p, p, sdf_capsule_grad(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.CYLINDER:
        p = sdf_cylinder(x_local, scale[0], scale[1], int(Axis.Z))
        return p, p, sdf_cylinder_grad(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.CONE:
        p = sdf_cone(x_local, scale[0], scale[1], int(Axis.Z))
        return p, p, sdf_cone_grad(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.ELLIPSOID:
        p = sdf_ellipsoid(x_local, scale)
        return p, p, sdf_ellipsoid_grad(x_local, scale)
    if geo == GeoType.PLANE:
        p = sdf_plane(x_local, scale[0] * 0.5, scale[1] * 0.5)
        return p, p, wp.vec3(0.0, 0.0, 1.0)

    # Volume SDF (mesh / convex / other). Honor the descriptor's scale_baked flag: if the shape
    # scale was baked into the grid (e.g. hydroelastic primitives), query directly in shape-local
    # (= scaled) space; otherwise apply query-time scaling (mirrors mesh_sdf_collision_kernel +
    # scale_sdf_result_to_world).
    tex = texture_sdf_table[shape_sdf_index]
    if tex.scale_baked:
        d, g = texture_sample_sdf_grad(tex, x_local)
        return d, d, g
    inv_scale = wp.vec3(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    dist, grad = texture_sample_sdf_grad(tex, wp.cw_div(x_local, scale))
    # texture_sample_sdf_grad's gradient is not unit-normalized, so normalize it before using it as a
    # surface normal (otherwise the stretch below is scaled by |grad|, corrupting the distance).
    grad_norm = wp.length(grad)
    if grad_norm > 0.0:
        grad = grad / grad_norm
    # Convert the normalized-frame distance to the shape-local frame. Under nonuniform scale the
    # displacement to the closest surface point stretches per axis, so the accurate local distance is
    # dist * |scale * grad| with grad the UNIT normal (the stretch ALONG the surface normal) -- exact
    # for an axis-aligned face, first-order near the surface where contacts live. min|scale| is a cheap
    # conservative lower bound for the cull/search. wp.length() / wp.min(wp.abs()) keep a mirrored
    # (negative) scale sign-correct; the mirror itself is applied by the cw_div query and by inv_scale.
    stretch = wp.length(wp.cw_mul(scale, grad))
    min_scale = wp.min(wp.abs(scale))
    scaled_grad = wp.cw_mul(grad, inv_scale)
    grad_len = wp.length(scaled_grad)
    if grad_len > 0.0:
        scaled_grad = scaled_grad / grad_len
    else:
        scaled_grad = grad
    return dist * min_scale, dist * stretch, scaled_grad


@wp.func
def optimize_edge_sdf(
    geo: wp.int32,
    scale: wp.vec3,
    p: wp.vec3,
    q: wp.vec3,
    shape_sdf_index: wp.int32,
    texture_sdf_table: wp.array[TextureSDFData],
    n_iter: wp.int32,
):
    """argmin_{u in [0,1]} phi((1-u) p + u q) by golden-section search (Macklin sec. 4).

    Fixed ``n_iter`` iterations -> graph-capturable. Returns ``(u, x_local, phi, grad)`` at the
    minimizing point. Also used as the line search inside :func:`optimize_face_sdf`.
    """
    inv_phi = float(0.6180339887498949)  # 1 / golden ratio
    lo = float(0.0)
    hi = float(1.0)
    c = hi - (hi - lo) * inv_phi
    d = lo + (hi - lo) * inv_phi
    fc, _fc_a, _gc = eval_shape_sdf(geo, scale, (1.0 - c) * p + c * q, shape_sdf_index, texture_sdf_table)
    fd, _fd_a, _gd = eval_shape_sdf(geo, scale, (1.0 - d) * p + d * q, shape_sdf_index, texture_sdf_table)
    for _i in range(n_iter):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - (hi - lo) * inv_phi
            fc, _fc_a, _gc = eval_shape_sdf(geo, scale, (1.0 - c) * p + c * q, shape_sdf_index, texture_sdf_table)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + (hi - lo) * inv_phi
            fd, _fd_a, _gd = eval_shape_sdf(geo, scale, (1.0 - d) * p + d * q, shape_sdf_index, texture_sdf_table)
    u = 0.5 * (lo + hi)
    x = (1.0 - u) * p + u * q
    _phi_l, phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
    return u, x, phi, grad


@wp.func
def optimize_face_sdf(
    geo: wp.int32,
    scale: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_sdf_index: wp.int32,
    texture_sdf_table: wp.array[TextureSDFData],
    n_iter: wp.int32,
    ls_iter: wp.int32,
):
    """argmin phi over the soft triangle by Frank-Wolfe on the barycentric simplex (Macklin sec. 3).

    Each step picks the simplex vertex minimizing the linearized objective ``grad . corner`` (eq. 4)
    and line-searches phi toward it with :func:`optimize_edge_sdf`. Fixed ``n_iter`` / ``ls_iter``
    iterations -> graph-capturable. Returns ``(bary, x_local, phi, grad)``.
    """
    # Start at the centroid (interior). A corner start can strand Frank-Wolfe on a simplex edge
    # for non-smooth fields (e.g. a box-corner ridge), because the analytic gradient is single-axis
    # and never selects the third vertex. From the centroid, FW can move toward any vertex.
    bary = wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    for _i in range(n_iter):
        x = bary[0] * a + bary[1] * b + bary[2] * c
        _phi_l, _phi_x, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
        # Frank-Wolfe vertex: argmin_k grad . corner_k (Macklin eq. 4).
        da = wp.dot(grad, a)
        db = wp.dot(grad, b)
        dc = wp.dot(grad, c)
        s = wp.vec3(1.0, 0.0, 0.0)
        if db <= da and db <= dc:
            s = wp.vec3(0.0, 1.0, 0.0)
        elif dc <= da and dc <= db:
            s = wp.vec3(0.0, 0.0, 1.0)
        target = s[0] * a + s[1] * b + s[2] * c
        gamma, _lx, _lphi, _lgrad = optimize_edge_sdf(
            geo, scale, x, target, shape_sdf_index, texture_sdf_table, ls_iter
        )
        bary = (1.0 - gamma) * bary + gamma * s

    x = bary[0] * a + bary[1] * b + bary[2] * c
    _phi_l, phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
    return bary, x, phi, grad


@wp.func
def _shape_frames(
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_index: wp.int32,
):
    """Return (X_bs, X_ws, X_sw): shape-local->body, shape-local->world, world->shape-local."""
    rigid_body = shape_body[shape_index]
    X_wb = wp.transform_identity()
    if rigid_body >= 0:
        X_wb = body_q[rigid_body]
    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)
    return X_bs, X_ws, X_sw


@wp.func
def _emit_soft_ef_contact(
    tid: wp.int32,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    corners: wp.vec3i,
    bary: wp.vec3,
    shape_index: wp.int32,
    body_pos: wp.vec3,
    body_vel: wp.vec3,
    normal: wp.vec3,
):
    """Append one edge/face record into the single unified soft-contact stream.

    Uses :func:`counter_increment` on the shared soft counter so the chosen index is recorded per
    thread (in ``soft_contact_tids``) for differentiable backward replay, matching the legacy
    particle pass. ``tid_base`` offsets this pass's thread ids into the shared tids array so the
    three passes (particle / edge / face) never alias the same tids slot: particle uses ``[0, ...)``,
    edge ``[n_particle_pairs, ...)``, face ``[n_particle_pairs + n_edge_pairs, ...)``. The offsets
    are static (pair counts fixed at pipeline init), so this stays CUDA-graph-capturable.

    The counter is incremented even when the record overflows ``soft_contact_max`` (the write is
    guarded); ``counter_increment`` returns -1 in that case."""
    idx = counter_increment(soft_contact_count, 0, soft_contact_tids, tid + tid_base, soft_contact_max)
    if idx >= 0:
        soft_contact_particle[idx] = -1  # edge/face record: no single particle id
        soft_contact_indices[idx] = corners
        soft_contact_barycentric[idx] = bary
        soft_contact_shape[idx] = shape_index
        soft_contact_body_pos[idx] = body_pos
        soft_contact_body_vel[idx] = body_vel
        soft_contact_normal[idx] = normal


@wp.kernel
def create_soft_face_contacts(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    sdf_face_iters: wp.int32,
    sdf_ls_iters: wp.int32,
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """One thread per world-compatible (soft triangle, shape) pair. Minimizes the rigid SDF over the
    triangle interior and emits a unified ``(v0, v1, v2)`` face record if within margin. Pairs are
    precomputed world-filtered (like ``soft_rigid_contact_pairs``), so no per-thread world check is
    needed. ``tid_base`` is n_particle_pairs + n_edge_pairs (this pass's offset into the shared
    replay-tids array)."""
    tid = wp.tid()
    pair = face_pairs[tid]
    t = pair[0]
    shape_index = pair[1]
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    geo = shape_type[shape_index]
    sdf_idx = shape_sdf_index[shape_index]
    if (not _is_analytic(geo)) and sdf_idx < 0:
        # Mesh without a provisioned SDF: the legacy per-particle path still covers it, and the
        # pipeline already warned once about this at construction.
        return

    a_idx = tri_indices[t, 0]
    b_idx = tri_indices[t, 1]
    c_idx = tri_indices[t, 2]
    radius = wp.max(particle_radius[a_idx], wp.max(particle_radius[b_idx], particle_radius[c_idx]))

    # _s suffix = shape-local frame (matching the X_*s transforms: b = body, w = world, s = shape).
    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    a_s = wp.transform_point(X_sw, particle_q[a_idx])
    b_s = wp.transform_point(X_sw, particle_q[b_idx])
    c_s = wp.transform_point(X_sw, particle_q[c_idx])
    scale = shape_scale[shape_index]
    # Per-shape contact margin (#2994), same threshold term as the legacy particle pass.
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + s_margin + radius

    centroid_s = (a_s + b_s + c_s) / 3.0
    phi_c, _phi_c_a, _grad_c = eval_shape_sdf(geo, scale, centroid_s, sdf_idx, texture_sdf_table)
    # Conservative cull: the SDF is ~1-Lipschitz, so the triangle's minimum is >= phi_c minus the
    # farthest centroid-to-point distance, which is always a vertex. circumradius can be smaller than
    # that for non-equilateral triangles (e.g. 3-4-5: R=2.5 vs 2.85) and would drop valid contacts.
    reach = wp.max(wp.length(a_s - centroid_s), wp.max(wp.length(b_s - centroid_s), wp.length(c_s - centroid_s)))
    if phi_c > threshold + reach:
        return

    bary, x, phi, grad = optimize_face_sdf(
        geo, scale, a_s, b_s, c_s, sdf_idx, texture_sdf_table, sdf_face_iters, sdf_ls_iters
    )
    if phi < threshold:
        y = x - phi * grad
        _emit_soft_ef_contact(
            tid,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            wp.vec3i(a_idx, b_idx, c_idx),
            bary,
            shape_index,
            wp.transform_point(X_bs, y),
            wp.vec3(0.0, 0.0, 0.0),
            wp.transform_vector(X_ws, grad),
        )


@wp.kernel
def create_soft_edge_contacts(
    edge_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    sdf_edge_iters: wp.int32,
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """One thread per world-compatible (unique soft edge, shape) pair. Minimizes the rigid SDF along
    the edge and emits a unified ``(v0, v1, -1)`` edge record if within margin. The endpoints come
    straight from ``edge_indices[e, 2:4]`` -- no triangle attribution needed. Unique edges ->
    structural dedup. Pairs are precomputed world-filtered, so no per-thread world check is needed.
    ``tid_base`` is n_particle_pairs (this pass's offset into the shared replay-tids array)."""
    tid = wp.tid()
    pair = edge_pairs[tid]
    e = pair[0]
    shape_index = pair[1]

    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    geo = shape_type[shape_index]
    sdf_idx = shape_sdf_index[shape_index]
    if (not _is_analytic(geo)) and sdf_idx < 0:
        return

    # edge_indices rows are [o0, o1, v0, v1]; cols 2,3 are the edge's endpoint particle ids.
    v0 = edge_indices[e, 2]
    v1 = edge_indices[e, 3]
    radius = wp.max(particle_radius[v0], particle_radius[v1])

    # _s suffix = shape-local frame (matching the X_*s transforms: b = body, w = world, s = shape).
    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    p_s = wp.transform_point(X_sw, particle_q[v0])
    q_s = wp.transform_point(X_sw, particle_q[v1])
    scale = shape_scale[shape_index]
    # Per-shape contact margin (#2994), same threshold term as the legacy particle pass.
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + s_margin + radius

    mid_s = 0.5 * (p_s + q_s)
    phi_m, _phi_m_a, _grad_m = eval_shape_sdf(geo, scale, mid_s, sdf_idx, texture_sdf_table)
    if phi_m > threshold + 0.5 * wp.length(q_s - p_s):
        return

    u, x, phi, grad = optimize_edge_sdf(geo, scale, p_s, q_s, sdf_idx, texture_sdf_table, sdf_edge_iters)
    if phi < threshold:
        y = x - phi * grad
        # optimize_edge_sdf parameterizes x = (1 - u) * p_s + u * q_s, so v0 carries weight 1 - u.
        _emit_soft_ef_contact(
            tid,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            wp.vec3i(v0, v1, -1),
            wp.vec3(1.0 - u, u, 0.0),
            shape_index,
            wp.transform_point(X_bs, y),
            wp.vec3(0.0, 0.0, 0.0),
            wp.transform_vector(X_ws, grad),
        )


def launch_soft_ef_contacts(*, model, state, contacts, margin: float, device, edge_pairs, face_pairs, n_particle_pairs):
    """Launch the soft EDGE and FACE passes (the soft-particle pass is the legacy kernel).

    ``edge_pairs`` / ``face_pairs`` are precomputed world-compatible (soft feature, shape) index
    pairs (``wp.vec2i``), analogous to :func:`_build_soft_particle_rigid_contact_pairs` for the
    particle pass -- one thread per pair, so cross-world features never reach the kernel. Edge
    endpoints come straight from ``model.edge_indices``; no mesh adjacency is needed.

    ``n_particle_pairs`` is the particle pass's launch dim; combined with the edge-pair count it
    forms the static per-pass offsets into the shared replay-tids array, so the three passes never
    alias a tids slot. All passes share one soft counter, so launch order is immaterial (records
    self-describe via -1 padding; nothing reads by range)."""
    # Nothing to do when there are no candidate pairs -- covers the no-soft-mesh, no-shape, and
    # flag-off cases, since the pairs are built from the mesh edges and shapes.
    n_edge_pairs = int(edge_pairs.shape[0])
    n_face_pairs = int(face_pairs.shape[0])
    if n_edge_pairs == 0 and n_face_pairs == 0:
        return

    shape_args = [
        model.shape_body,
        model.shape_type,
        model.shape_flags,
        model.shape_transform,
        model.shape_scale,
        state.body_q,
        model._shape_sdf_index,
        model._texture_sdf_data,
        model.shape_margin,
    ]
    outputs = [
        contacts.soft_contact_count,
        contacts.soft_contact_tids,
        contacts.soft_contact_particle,
        contacts.soft_contact_indices,
        contacts.soft_contact_barycentric,
        contacts.soft_contact_shape,
        contacts.soft_contact_body_pos,
        contacts.soft_contact_body_vel,
        contacts.soft_contact_normal,
    ]

    if n_edge_pairs > 0:
        wp.launch(
            create_soft_edge_contacts,
            dim=n_edge_pairs,
            inputs=[
                edge_pairs,
                state.particle_q,
                model.particle_radius,
                model.edge_indices,
                *shape_args,
                SDF_EDGE_ITERS,
                margin,
                n_particle_pairs,
                contacts.soft_contact_max,
            ],
            outputs=outputs,
            device=device,
        )
    if n_face_pairs > 0:
        wp.launch(
            create_soft_face_contacts,
            dim=n_face_pairs,
            inputs=[
                face_pairs,
                state.particle_q,
                model.particle_radius,
                model.tri_indices,
                *shape_args,
                SDF_FACE_ITERS,
                SDF_LS_ITERS,
                margin,
                n_particle_pairs + n_edge_pairs,
                contacts.soft_contact_max,
            ],
            outputs=outputs,
            device=device,
        )
