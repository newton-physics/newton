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

from __future__ import annotations

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
    sdf_sphere,
    sdf_sphere_grad,
    triangle_closest_point,
)
from .sdf_texture import TextureSDFData, texture_sample_sdf, texture_sample_sdf_grad
from .types import Axis, GeoType

# Fixed iteration counts -> data-independent loops -> CUDA-graph-capturable. Tuned against a
# brute-force grid reference
# (newton/tests/test_collision_pipeline.py, TestFullSurfaceSoftContact): edge golden-section is
# accurate (~1e-4); the face
# Frank-Wolfe tail is ~O(1/iters) (~3e-3 at 24 iters), sufficient for contact within margin.
SDF_EDGE_ITERS = 24
SDF_FACE_ITERS = 24
SDF_LS_ITERS = 16

# Keep smaller workloads fused, where rematerializing compacted features costs more than it saves.
_SDF_COMPACTION_MIN_PAIRS = 2 * 1024 * 1024
_SDF_FACE_CLASSIFY_BLOCK_DIM = 128
_SDF_SPECIALIZATION_MIN_PAIRS_PER_GEO = 1024 * 1024
_SDF_EDGE_MIXED_SPECIALIZATION_MIN_PAIRS_PER_GEO = 5 * 1024 * 1024
# Bound the compact optimization grid while keeping enough blocks to fill large GPUs.
_SDF_FALLBACK_GRID_SIZE = 262144


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
def _eval_plane_sdf(scale: wp.vec3, point: wp.vec3):
    """Return sheet distance and normal, with negative Z retained inside finite footprints."""
    half_width = 0.5 * scale[0]
    half_length = 0.5 * scale[1]
    if half_width <= 0.0 or half_length <= 0.0:
        distance = point[2]
        return distance, distance, wp.vec3(0.0, 0.0, 1.0)

    # Within the finite footprint, preserve the plane's signed distance. Outside it, use the
    # Euclidean distance to the open sheet, including the correct edge/corner gradient.
    if wp.abs(point[0]) <= half_width and wp.abs(point[1]) <= half_length:
        distance = point[2]
        return distance, distance, wp.vec3(0.0, 0.0, 1.0)

    closest = wp.vec3(
        wp.clamp(point[0], -half_width, half_width),
        wp.clamp(point[1], -half_length, half_length),
        0.0,
    )
    delta = point - closest
    distance = wp.length(delta)
    grad = wp.vec3(0.0, 0.0, 1.0)
    if distance > 0.0:
        grad = delta / distance
    return distance, distance, grad


@wp.func
def _plane_corner(scale: wp.vec3, corner: int) -> wp.vec3:
    x = 0.5 * scale[0]
    y = 0.5 * scale[1]
    if corner == 0 or corner == 3:
        x = -x
    if corner < 2:
        y = -y
    return wp.vec3(x, y, 0.0)


@wp.func
def _closest_edge_plane(scale: wp.vec3, p: wp.vec3, q: wp.vec3):
    """Minimize the signed sheet distance using footprint clipping and exact boundary features."""
    u = float(0.0)
    phi, _phi, grad = _eval_plane_sdf(scale, p)
    phi_q, _phi_q, grad_q = _eval_plane_sdf(scale, q)
    if phi_q < phi:
        u, phi, grad = 1.0, phi_q, grad_q
    if scale[0] > 0.0 and scale[1] > 0.0:
        edge = q - p
        lo = float(0.0)
        hi = float(1.0)
        # Below the footprint the signed distance is linear in Z; its minimum is at a clipped end.
        for axis in range(2):
            half = 0.5 * scale[axis]
            if edge[axis] != 0.0:
                t0 = (-half - p[axis]) / edge[axis]
                t1 = (half - p[axis]) / edge[axis]
                lo = wp.max(lo, wp.min(t0, t1))
                hi = wp.min(hi, wp.max(t0, t1))
            elif wp.abs(p[axis]) > half:
                hi = -1.0
        if lo <= hi:
            t = lo
            if edge[2] < 0.0:
                t = hi
            z = p[2] + t * edge[2]
            if z < phi:
                u, phi, grad = t, z, wp.vec3(0.0, 0.0, 1.0)
        for corner in range(4):
            a = _plane_corner(scale, corner)
            b = _plane_corner(scale, (corner + 1) % 4)
            st = wp.closest_point_edge_edge(p, q, a, b, 0.0)
            delta = p + st[0] * edge - (a + st[1] * (b - a))
            distance = wp.length(delta)
            if distance < phi:
                u, phi = st[0], distance
                grad = wp.vec3(0.0, 0.0, 1.0)
                if distance > 0.0:
                    grad = delta / distance
    return u, (1.0 - u) * p + u * q, phi, grad


@wp.func
def _closest_face_plane(scale: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Minimize over triangle edges, quad vertices, and vertices of the clipped footprint."""
    u, x, phi, grad = _closest_edge_plane(scale, a, b)
    bary = wp.vec3(1.0 - u, u, 0.0)
    for edge in range(2):
        p, q = b, c
        if edge == 1:
            p, q = c, a
        t, point, distance, normal = _closest_edge_plane(scale, p, q)
        if distance < phi:
            x, phi, grad = point, distance, normal
            bary = wp.vec3(0.0, 1.0 - t, t)
            if edge == 1:
                bary = wp.vec3(t, 0.0, 1.0 - t)
    if scale[0] > 0.0 and scale[1] > 0.0:
        ab, ac = b - a, c - a
        determinant = ab[0] * ac[1] - ab[1] * ac[0]
        for corner in range(4):
            y = _plane_corner(scale, corner)
            point, weights, _feature = triangle_closest_point(a, b, c, y)
            delta = point - y
            distance = wp.length(delta)
            if distance < phi:
                x, bary, phi = point, weights, distance
                grad = wp.vec3(0.0, 0.0, 1.0)
                if distance > 0.0:
                    grad = delta / distance
            # Remaining clipped-polygon vertices lie above/below a quad corner. Intersect its
            # vertical line with the soft face; perpendicular closest points alone miss penetration.
            if determinant != 0.0:
                offset = y - a
                v = (offset[0] * ac[1] - offset[1] * ac[0]) / determinant
                w = (ab[0] * offset[1] - ab[1] * offset[0]) / determinant
                if v >= 0.0 and w >= 0.0 and v + w <= 1.0:
                    weights = wp.vec3(1.0 - v - w, v, w)
                    point = weights[0] * a + weights[1] * b + weights[2] * c
                    if point[2] < phi:
                        x, bary, phi, grad = point, weights, point[2], wp.vec3(0.0, 0.0, 1.0)
    return bary, x, phi, grad


@wp.func
def _eval_shape_sdf_lower(
    geo: wp.int32,
    scale: wp.vec3,
    x_local: wp.vec3,
    shape_sdf_index: wp.int32,
    texture_sdf_table: wp.array[TextureSDFData],
):
    """Return only the conservative SDF value needed by culls and line searches."""
    if geo == GeoType.SPHERE:
        return sdf_sphere(x_local, scale[0])
    if geo == GeoType.BOX:
        return sdf_box(x_local, scale[0], scale[1], scale[2])
    if geo == GeoType.CAPSULE:
        return sdf_capsule(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.CYLINDER:
        return sdf_cylinder(x_local, scale[0], scale[1], int(Axis.Z), -1.0, scale[2])
    if geo == GeoType.CONE:
        return sdf_cone(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.ELLIPSOID:
        return sdf_ellipsoid(x_local, scale)
    if geo == GeoType.PLANE:
        phi_lower, _phi, _grad = _eval_plane_sdf(scale, x_local)
        return phi_lower

    tex = texture_sdf_table[shape_sdf_index]
    if tex.scale_baked:
        return texture_sample_sdf(tex, x_local)
    dist = texture_sample_sdf(tex, wp.cw_div(x_local, scale))
    return dist * wp.min(wp.abs(scale))


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
        p = sdf_cylinder(x_local, scale[0], scale[1], int(Axis.Z), -1.0, scale[2])
        return p, p, sdf_cylinder_grad(x_local, scale[0], scale[1], int(Axis.Z), -1.0, scale[2])
    if geo == GeoType.CONE:
        p = sdf_cone(x_local, scale[0], scale[1], int(Axis.Z))
        return p, p, sdf_cone_grad(x_local, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.ELLIPSOID:
        p = sdf_ellipsoid(x_local, scale)
        return p, p, sdf_ellipsoid_grad(x_local, scale)
    if geo == GeoType.PLANE:
        return _eval_plane_sdf(scale, x_local)

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
    Finite planes use direct feature queries in the contact kernels instead of this search.
    """
    inv_phi = float(0.6180339887498949)  # 1 / golden ratio
    lo = float(0.0)
    hi = float(1.0)
    c = hi - (hi - lo) * inv_phi
    d = lo + (hi - lo) * inv_phi
    fc = _eval_shape_sdf_lower(geo, scale, (1.0 - c) * p + c * q, shape_sdf_index, texture_sdf_table)
    fd = _eval_shape_sdf_lower(geo, scale, (1.0 - d) * p + d * q, shape_sdf_index, texture_sdf_table)
    for _i in range(n_iter):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - (hi - lo) * inv_phi
            fc = _eval_shape_sdf_lower(geo, scale, (1.0 - c) * p + c * q, shape_sdf_index, texture_sdf_table)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + (hi - lo) * inv_phi
            fd = _eval_shape_sdf_lower(geo, scale, (1.0 - d) * p + d * q, shape_sdf_index, texture_sdf_table)
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
    x = bary[0] * a + bary[1] * b + bary[2] * c
    _phi_l, phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)

    # Reusing the line-search endpoint saves one SDF evaluation per outer iteration for analytic
    # shapes. Keep the volume-SDF path's shorter-lived texture state; carrying it across iterations
    # reduced occupancy in representative mesh workloads.
    reuse_line_result = _is_analytic(geo)
    for _i in range(n_iter):
        if not reuse_line_result and _i > 0:
            x = bary[0] * a + bary[1] * b + bary[2] * c
            _phi_l, phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)

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
        gamma, line_x, line_phi, line_grad = optimize_edge_sdf(
            geo, scale, x, target, shape_sdf_index, texture_sdf_table, ls_iter
        )
        bary = (1.0 - gamma) * bary + gamma * s
        if reuse_line_result:
            x = line_x
            phi = line_phi
            grad = line_grad

    if not reuse_line_result:
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
def _soft_feature_aabb_misses_shape(
    geo: wp.int32,
    shape_index: wp.int32,
    shape_gap: wp.array[float],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    feature_lower: wp.vec3,
    feature_upper: wp.vec3,
    margin: float,
    radius: float,
) -> bool:
    """Return whether an expanded soft-feature AABB is disjoint from the rigid-shape AABB.

    Planes bypass the test (their finite stored bounds cannot represent an infinite contact
    surface); an empty AABB array disables the rejection for isolated/reference callers.
    """
    if shape_aabb_lower.shape[0] == 0 or geo == GeoType.PLANE:
        return False

    gap = shape_gap[shape_index]
    expansion = margin + radius + wp.max(0.0, -gap)
    expansion_vec = wp.vec3(expansion, expansion, expansion)
    lower = feature_lower - expansion_vec
    upper = feature_upper + expansion_vec
    rigid_lower = shape_aabb_lower[shape_index]
    rigid_upper = shape_aabb_upper[shape_index]
    return (
        upper[0] < rigid_lower[0]
        or upper[1] < rigid_lower[1]
        or upper[2] < rigid_lower[2]
        or lower[0] > rigid_upper[0]
        or lower[1] > rigid_upper[1]
        or lower[2] > rigid_upper[2]
    )


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


def _create_soft_face_contact_kernel(compact_sdf: bool):
    module_name = f"soft_face_contacts_{compact_sdf}"

    @wp.kernel(module=module_name)
    def soft_face_contact_kernel(
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
        shape_gap: wp.array[float],
        shape_aabb_lower: wp.array[wp.vec3],
        shape_aabb_upper: wp.array[wp.vec3],
        margin: float,
        tid_base: wp.int32,
        soft_contact_max: wp.int32,
        fallback_tids: wp.array[wp.int32],
        fallback_count: wp.array[wp.int32],
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
        """Cull face pairs, emit exact sphere contacts, and compact SDF optimization work."""
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

        a_w = particle_q[a_idx]
        b_w = particle_q[b_idx]
        c_w = particle_q[c_idx]
        feature_lower = wp.vec3(
            wp.min(a_w[0], wp.min(b_w[0], c_w[0])),
            wp.min(a_w[1], wp.min(b_w[1], c_w[1])),
            wp.min(a_w[2], wp.min(b_w[2], c_w[2])),
        )
        feature_upper = wp.vec3(
            wp.max(a_w[0], wp.max(b_w[0], c_w[0])),
            wp.max(a_w[1], wp.max(b_w[1], c_w[1])),
            wp.max(a_w[2], wp.max(b_w[2], c_w[2])),
        )
        if _soft_feature_aabb_misses_shape(
            geo,
            shape_index,
            shape_gap,
            shape_aabb_lower,
            shape_aabb_upper,
            feature_lower,
            feature_upper,
            margin,
            radius,
        ):
            return

        # _s suffix = shape-local frame (matching the X_*s transforms: b = body, w = world, s = shape).
        X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
        a_s = wp.transform_point(X_sw, a_w)
        b_s = wp.transform_point(X_sw, b_w)
        c_s = wp.transform_point(X_sw, c_w)
        scale = shape_scale[shape_index]
        # Per-shape contact margin (#2994), same threshold term as the legacy particle pass.
        s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
        threshold = margin + s_margin + radius

        centroid_s = (a_s + b_s + c_s) / 3.0
        if wp.static(compact_sdf):
            phi_c = _eval_shape_sdf_lower(geo, scale, centroid_s, sdf_idx, texture_sdf_table)
        else:
            phi_c, _phi_c_a, _grad_c = eval_shape_sdf(geo, scale, centroid_s, sdf_idx, texture_sdf_table)
        # Conservative cull: the SDF is ~1-Lipschitz, so the triangle's minimum is >= phi_c minus the
        # farthest centroid-to-point distance, which is always a vertex. circumradius can be smaller than
        # that for non-equilateral triangles (e.g. 3-4-5: R=2.5 vs 2.85) and would drop valid contacts.
        reach = wp.max(wp.length(a_s - centroid_s), wp.max(wp.length(b_s - centroid_s), wp.length(c_s - centroid_s)))
        # Finite sheets change sign at the footprint boundary and are not Lipschitz below it.
        if geo != GeoType.PLANE and phi_c > threshold + reach:
            return

        bary = wp.vec3(0.0)
        x = wp.vec3(0.0)
        phi = float(0.0)
        grad = wp.vec3(0.0)
        if geo == GeoType.SPHERE:
            x, bary, _feature = triangle_closest_point(a_s, b_s, c_s, wp.vec3(0.0))
            phi = sdf_sphere(x, scale[0])
            grad = sdf_sphere_grad(x, scale[0])
        elif geo == GeoType.PLANE and (scale[0] <= 0.0 or scale[1] <= 0.0):
            # The infinite-plane SDF is linear, so its minimum over a triangle is a vertex.
            bary = wp.vec3(1.0, 0.0, 0.0)
            x = a_s
            if b_s[2] <= a_s[2] and b_s[2] <= c_s[2]:
                bary = wp.vec3(0.0, 1.0, 0.0)
                x = b_s
            elif c_s[2] <= a_s[2] and c_s[2] <= b_s[2]:
                bary = wp.vec3(0.0, 0.0, 1.0)
                x = c_s
            phi = x[2]
            grad = wp.vec3(0.0, 0.0, 1.0)
        elif geo == GeoType.PLANE:
            bary, x, phi, grad = _closest_face_plane(scale, a_s, b_s, c_s)
        else:
            if wp.static(compact_sdf):
                fallback_slot = wp.atomic_add(fallback_count, 0, 1)
                fallback_tids[fallback_slot] = tid
                return
            bary, x, phi, grad = optimize_face_sdf(
                geo, scale, a_s, b_s, c_s, sdf_idx, texture_sdf_table, SDF_FACE_ITERS, SDF_LS_ITERS
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

    return soft_face_contact_kernel


create_soft_face_contacts = _create_soft_face_contact_kernel(False)
classify_soft_face_contacts = _create_soft_face_contact_kernel(True)


def _create_soft_face_sdf_contact_kernel(geo_filter: int):
    module_name = f"soft_face_sdf_contacts_{geo_filter}"

    @wp.kernel(module=module_name)
    def create_soft_face_sdf_contacts(
        fallback_tids: wp.array[wp.int32],
        fallback_count: wp.array[wp.int32],
        fallback_grid_size: wp.int32,
        face_pairs: wp.array[wp.vec2i],
        particle_q: wp.array[wp.vec3],
        particle_radius: wp.array[float],
        tri_indices: wp.array2d[wp.int32],
        shape_body: wp.array[wp.int32],
        shape_type: wp.array[wp.int32],
        shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        shape_sdf_index: wp.array[wp.int32],
        texture_sdf_table: wp.array[TextureSDFData],
        shape_margin: wp.array[float],
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
        """Optimize compacted face pairs other than spheres and planes."""
        offset = wp.tid()
        count = wp.min(fallback_count[0], fallback_tids.shape[0])
        for fallback_slot in range(offset, count, fallback_grid_size):
            source_tid = fallback_tids[fallback_slot]
            pair = face_pairs[source_tid]
            t = pair[0]
            shape_index = pair[1]

            a_idx = tri_indices[t, 0]
            b_idx = tri_indices[t, 1]
            c_idx = tri_indices[t, 2]
            radius = wp.max(particle_radius[a_idx], wp.max(particle_radius[b_idx], particle_radius[c_idx]))

            X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
            a_s = wp.transform_point(X_sw, particle_q[a_idx])
            b_s = wp.transform_point(X_sw, particle_q[b_idx])
            c_s = wp.transform_point(X_sw, particle_q[c_idx])
            scale = shape_scale[shape_index]
            s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
            threshold = margin + s_margin + radius
            pair_geo = shape_type[shape_index]
            if wp.static(geo_filter >= 0):
                if pair_geo != wp.static(geo_filter):
                    continue
                geo = wp.static(geo_filter)
            else:
                geo = pair_geo
            sdf_idx = shape_sdf_index[shape_index]
            bary, x, phi, grad = optimize_face_sdf(
                geo, scale, a_s, b_s, c_s, sdf_idx, texture_sdf_table, SDF_FACE_ITERS, SDF_LS_ITERS
            )
            if phi < threshold:
                y = x - phi * grad
                _emit_soft_ef_contact(
                    source_tid,
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
                    wp.vec3(0.0),
                    wp.transform_vector(X_ws, grad),
                )

    return create_soft_face_sdf_contacts


# The common face-pair stream contains only analytic shapes. Specializing its compact fallback
# removes the runtime SDF dispatch from every inner Frank-Wolfe/line-search iteration.
create_soft_face_sdf_contacts = _create_soft_face_sdf_contact_kernel(-1)
_SDF_SPECIALIZED_GEO_TYPES = (
    GeoType.BOX,
    GeoType.CAPSULE,
    GeoType.CYLINDER,
    GeoType.CONE,
    GeoType.ELLIPSOID,
)
_SDF_FACE_KERNEL_BY_GEO = {geo: _create_soft_face_sdf_contact_kernel(int(geo)) for geo in _SDF_SPECIALIZED_GEO_TYPES}


def _create_soft_edge_contact_kernel(compact_sdf: bool):
    module_name = f"soft_edge_contacts_{compact_sdf}"

    @wp.kernel(module=module_name)
    def soft_edge_contact_kernel(
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
        shape_gap: wp.array[float],
        shape_aabb_lower: wp.array[wp.vec3],
        shape_aabb_upper: wp.array[wp.vec3],
        sdf_edge_iters: wp.int32,
        margin: float,
        tid_base: wp.int32,
        soft_contact_max: wp.int32,
        fallback_tids: wp.array[wp.int32],
        fallback_count: wp.array[wp.int32],
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

        p_w = particle_q[v0]
        q_w = particle_q[v1]
        feature_lower = wp.vec3(
            wp.min(p_w[0], q_w[0]),
            wp.min(p_w[1], q_w[1]),
            wp.min(p_w[2], q_w[2]),
        )
        feature_upper = wp.vec3(
            wp.max(p_w[0], q_w[0]),
            wp.max(p_w[1], q_w[1]),
            wp.max(p_w[2], q_w[2]),
        )
        if _soft_feature_aabb_misses_shape(
            geo,
            shape_index,
            shape_gap,
            shape_aabb_lower,
            shape_aabb_upper,
            feature_lower,
            feature_upper,
            margin,
            radius,
        ):
            return

        # _s suffix = shape-local frame (matching the X_*s transforms: b = body, w = world, s = shape).
        X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
        p_s = wp.transform_point(X_sw, p_w)
        q_s = wp.transform_point(X_sw, q_w)
        scale = shape_scale[shape_index]
        # Per-shape contact margin (#2994), same threshold term as the legacy particle pass.
        s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
        threshold = margin + s_margin + radius

        mid_s = 0.5 * (p_s + q_s)
        phi_m = _eval_shape_sdf_lower(geo, scale, mid_s, sdf_idx, texture_sdf_table)
        if geo != GeoType.PLANE and phi_m > threshold + 0.5 * wp.length(q_s - p_s):
            return

        if wp.static(compact_sdf) and geo != GeoType.SPHERE and geo != GeoType.PLANE:
            fallback_slot = wp.atomic_add(fallback_count, 0, 1)
            fallback_tids[fallback_slot] = tid
            return

        if geo == GeoType.SPHERE:
            edge = q_s - p_s
            edge_len_sq = wp.length_sq(edge)
            u = float(0.0)
            if edge_len_sq > 0.0:
                u = wp.clamp(-wp.dot(p_s, edge) / edge_len_sq, 0.0, 1.0)
            x = p_s + u * edge
            phi = sdf_sphere(x, scale[0])
            grad = sdf_sphere_grad(x, scale[0])
        elif geo == GeoType.PLANE:
            u, x, phi, grad = _closest_edge_plane(scale, p_s, q_s)
        else:
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

    return soft_edge_contact_kernel


create_soft_edge_contacts = _create_soft_edge_contact_kernel(False)
classify_soft_edge_contacts = _create_soft_edge_contact_kernel(True)


def _create_soft_edge_sdf_contact_kernel(geo_filter: int):
    module_name = f"soft_edge_sdf_contacts_{geo_filter}"

    @wp.kernel(module=module_name)
    def create_soft_edge_sdf_contacts(
        fallback_tids: wp.array[wp.int32],
        fallback_count: wp.array[wp.int32],
        fallback_grid_size: wp.int32,
        edge_pairs: wp.array[wp.vec2i],
        particle_q: wp.array[wp.vec3],
        particle_radius: wp.array[float],
        edge_indices: wp.array2d[wp.int32],
        shape_body: wp.array[wp.int32],
        shape_type: wp.array[wp.int32],
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
        """Optimize compacted edge pairs other than spheres and planes."""
        offset = wp.tid()
        count = wp.min(fallback_count[0], fallback_tids.shape[0])
        for fallback_slot in range(offset, count, fallback_grid_size):
            source_tid = fallback_tids[fallback_slot]
            pair = edge_pairs[source_tid]
            edge_index = pair[0]
            shape_index = pair[1]

            v0 = edge_indices[edge_index, 2]
            v1 = edge_indices[edge_index, 3]
            radius = wp.max(particle_radius[v0], particle_radius[v1])

            X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
            p_s = wp.transform_point(X_sw, particle_q[v0])
            q_s = wp.transform_point(X_sw, particle_q[v1])
            scale = shape_scale[shape_index]
            s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
            threshold = margin + s_margin + radius
            pair_geo = shape_type[shape_index]
            if wp.static(geo_filter >= 0):
                if pair_geo != wp.static(geo_filter):
                    continue
                geo = wp.static(geo_filter)
            else:
                geo = pair_geo
            sdf_idx = shape_sdf_index[shape_index]
            u, x, phi, grad = optimize_edge_sdf(geo, scale, p_s, q_s, sdf_idx, texture_sdf_table, sdf_edge_iters)
            if phi < threshold:
                y = x - phi * grad
                _emit_soft_ef_contact(
                    source_tid,
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
                    wp.vec3(0.0),
                    wp.transform_vector(X_ws, grad),
                )

    return create_soft_edge_sdf_contacts


create_soft_edge_sdf_contacts = _create_soft_edge_sdf_contact_kernel(-1)
_SDF_EDGE_KERNEL_BY_GEO = {geo: _create_soft_edge_sdf_contact_kernel(int(geo)) for geo in _SDF_SPECIALIZED_GEO_TYPES}


def launch_soft_ef_contacts(
    *,
    model,
    state,
    contacts,
    margin: float,
    device,
    edge_pairs: wp.array[wp.vec2i],
    face_pairs: wp.array[wp.vec2i],
    sdf_fallback_tids,
    sdf_fallback_count,
    n_particle_pairs: int,
    edge_sdf_geo_types=None,
    face_sdf_geo_types=None,
    shape_aabb_lower: wp.array[wp.vec3] | None = None,
    shape_aabb_upper: wp.array[wp.vec3] | None = None,
):
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

    # The iterative SDF kernels are register-heavy on CUDA; smaller blocks improve occupancy.
    device_is_cuda = wp.get_device(device).is_cuda
    sdf_block_dim = 128 if device_is_cuda else 256

    if shape_aabb_lower is None:
        # Isolated kernel tests can intentionally disable the broad rejection. Production collision
        # always supplies the current narrow-phase AABBs, so graph capture never allocates here.
        shape_aabb_lower = wp.empty(0, dtype=wp.vec3, device=device)
        shape_aabb_upper = wp.empty(0, dtype=wp.vec3, device=device)

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
        model.shape_gap,
        shape_aabb_lower,
        shape_aabb_upper,
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
        compact_edge_sdf = device_is_cuda and n_edge_pairs >= _SDF_COMPACTION_MIN_PAIRS
        if compact_edge_sdf:
            sdf_fallback_count.zero_()
        edge_kernel = classify_soft_edge_contacts if compact_edge_sdf else create_soft_edge_contacts
        wp.launch(
            edge_kernel,
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
            outputs=[sdf_fallback_tids, sdf_fallback_count, *outputs],
            device=device,
            block_dim=sdf_block_dim,
        )
        if compact_edge_sdf:
            fallback_grid_size = min(n_edge_pairs, _SDF_FALLBACK_GRID_SIZE)
            edge_sdf_kernels = (create_soft_edge_sdf_contacts,)
            edge_specialization_min_pairs = _SDF_SPECIALIZATION_MIN_PAIRS_PER_GEO
            if edge_sdf_geo_types and len(edge_sdf_geo_types) > 1:
                edge_specialization_min_pairs = _SDF_EDGE_MIXED_SPECIALIZATION_MIN_PAIRS_PER_GEO
            if edge_sdf_geo_types and n_edge_pairs >= len(edge_sdf_geo_types) * edge_specialization_min_pairs:
                edge_sdf_kernels = tuple(_SDF_EDGE_KERNEL_BY_GEO[geo] for geo in edge_sdf_geo_types)

            for edge_sdf_kernel in edge_sdf_kernels:
                wp.launch(
                    edge_sdf_kernel,
                    dim=fallback_grid_size,
                    inputs=[
                        sdf_fallback_tids,
                        sdf_fallback_count,
                        fallback_grid_size,
                        edge_pairs,
                        state.particle_q,
                        model.particle_radius,
                        model.edge_indices,
                        model.shape_body,
                        model.shape_type,
                        model.shape_transform,
                        model.shape_scale,
                        state.body_q,
                        model._shape_sdf_index,
                        model._texture_sdf_data,
                        model.shape_margin,
                        SDF_EDGE_ITERS,
                        margin,
                        n_particle_pairs,
                        contacts.soft_contact_max,
                    ],
                    outputs=outputs,
                    device=device,
                    block_dim=sdf_block_dim,
                )
    if n_face_pairs > 0:
        compact_sdf = device_is_cuda and n_face_pairs >= _SDF_COMPACTION_MIN_PAIRS
        if compact_sdf:
            sdf_fallback_count.zero_()
        face_kernel = classify_soft_face_contacts if compact_sdf else create_soft_face_contacts
        wp.launch(
            face_kernel,
            dim=n_face_pairs,
            inputs=[
                face_pairs,
                state.particle_q,
                model.particle_radius,
                model.tri_indices,
                *shape_args,
                margin,
                n_particle_pairs + n_edge_pairs,
                contacts.soft_contact_max,
            ],
            outputs=[sdf_fallback_tids, sdf_fallback_count, *outputs],
            device=device,
            block_dim=_SDF_FACE_CLASSIFY_BLOCK_DIM,
        )
        if not compact_sdf:
            return
        fallback_grid_size = min(n_face_pairs, _SDF_FALLBACK_GRID_SIZE)
        face_sdf_kernels = (create_soft_face_sdf_contacts,)
        if face_sdf_geo_types and n_face_pairs >= (len(face_sdf_geo_types) * _SDF_SPECIALIZATION_MIN_PAIRS_PER_GEO):
            face_sdf_kernels = tuple(_SDF_FACE_KERNEL_BY_GEO[geo] for geo in face_sdf_geo_types)

        for face_sdf_kernel in face_sdf_kernels:
            wp.launch(
                face_sdf_kernel,
                dim=fallback_grid_size,
                inputs=[
                    sdf_fallback_tids,
                    sdf_fallback_count,
                    fallback_grid_size,
                    face_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.tri_indices,
                    model.shape_body,
                    model.shape_type,
                    model.shape_transform,
                    model.shape_scale,
                    state.body_q,
                    model._shape_sdf_index,
                    model._texture_sdf_data,
                    model.shape_margin,
                    margin,
                    n_particle_pairs + n_edge_pairs,
                    contacts.soft_contact_max,
                ],
                outputs=outputs,
                device=device,
                block_dim=sdf_block_dim,
            )
