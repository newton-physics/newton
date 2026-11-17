# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-storage contact queries for overlapping ellipsoids."""

import warp as wp


@wp.func
def _support(radii: wp.vec3, direction: wp.vec3) -> wp.vec3:
    scaled = wp.cw_mul(radii, direction)
    return wp.cw_mul(radii, scaled) / wp.length(scaled)


@wp.func
def _support_pair(
    radii_a: wp.vec3, radii_b: wp.vec3, rotation: wp.quat, position: wp.vec3, direction: wp.vec3
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    point_a = _support(radii_a, direction)
    point_b = position + wp.quat_rotate(rotation, _support(radii_b, -wp.quat_rotate_inv(rotation, direction)))
    return point_a, point_b, point_a - point_b


@wp.func
def _curvature(radii: wp.vec3, normal: wp.vec3, tangent_u: wp.vec3, tangent_v: wp.vec3) -> tuple[float, float, float]:
    squared = wp.cw_mul(radii, radii)
    gradient = wp.cw_mul(squared, normal)
    inverse = 1.0 / wp.sqrt(wp.dot(gradient, normal))
    inverse_cube = inverse * inverse * inverse
    u = wp.dot(tangent_u, gradient)
    v = wp.dot(tangent_v, gradient)
    uu = wp.dot(tangent_u, wp.cw_mul(squared, tangent_u)) * inverse - u * u * inverse_cube
    uv = wp.dot(tangent_u, wp.cw_mul(squared, tangent_v)) * inverse - u * v * inverse_cube
    vv = wp.dot(tangent_v, wp.cw_mul(squared, tangent_v)) * inverse - v * v * inverse_cube
    return uu, uv, vv


# Warp before 1.18 has no function inlining option.
_contact_func = wp.func(inline=False) if tuple(int(v) for v in wp.__version__.split(".")[:2]) >= (1, 18) else wp.func


@_contact_func
def collide_ellipsoids(
    radii_a: wp.vec3, radii_b: wp.vec3, rotation_b: wp.quat, position_b: wp.vec3
) -> tuple[bool, float, wp.vec3, wp.vec3]:
    """Return a certified overlap or leave the pair to the general convex solver.

    Both witnesses and the midpoint are expressed in A's frame. A sphere uses
    three equal radii. The return values are success, signed distance, midpoint,
    and the unit normal from A to B. Separated and ill-conditioned pairs return
    failure rather than supplying provisional contacts.

    Minimize the support-plane depth on the unit sphere with a two-dimensional
    Newton step. A local minimum alone does not certify penetration depth.
    Interior tangent balls provide a lower bound, and the support plane gives
    an upper bound; accept only when these agree within the accuracy budget.
    Witnesses must also agree with the reported normal within that budget.
    """
    normal = wp.vec3(1.0, 0.0, 0.0)
    center = wp.vec3(0.0)
    scale = wp.max(wp.max(radii_a), wp.max(radii_b))
    if scale <= 0.0 or wp.min(radii_a) <= 1e-4 * scale or wp.min(radii_b) <= 1e-4 * scale:
        return False, 0.0, center, normal
    if wp.length(position_b) > wp.length(radii_a) + wp.length(radii_b):
        return False, 0.0, center, normal

    # Keep curvature and support products away from absolute small-direction
    # cutoffs and overflow. The final witnesses are converted back to meters.
    a = radii_a / scale
    b = radii_b / scale
    position = position_b / scale
    tolerance = wp.max(1.0, wp.length(position) + wp.max(wp.length(a), wp.length(b)))
    accuracy = 1e-6 * tolerance
    bound = float(1e30)
    for seed in range(14):
        direction = wp.vec3(0.0)
        if seed < 6:
            direction[seed // 2] = 1.0 if seed % 2 == 0 else -1.0
        elif seed < 12:
            direction[(seed - 6) // 2] = 1.0 if seed % 2 == 0 else -1.0
            direction = wp.quat_rotate(rotation_b, direction)
        else:
            direction = position
            if seed == 13:
                direction = -direction
        if wp.length_sq(direction) > 0.0:
            direction = wp.normalize(direction)
            _, _, delta = _support_pair(a, b, rotation_b, position, direction)
            depth = wp.dot(delta, direction)
            if depth < bound:
                bound = depth
                normal = direction
    if bound <= 0.0:
        return False, 0.0, center, normal

    for _iteration in range(16):
        point_a, point_b, delta = _support_pair(a, b, rotation_b, position, normal)
        depth = wp.dot(delta, normal)
        distance = wp.length(delta)
        if depth <= 0.0 or distance == 0.0:
            return False, 0.0, center, normal
        axis = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(normal[0]) > 0.7:
            axis = wp.vec3(0.0, 1.0, 0.0)
        tangent_u = wp.normalize(wp.cross(normal, axis))
        tangent_v = wp.cross(normal, tangent_u)
        auu, auv, avv = _curvature(a, normal, tangent_u, tangent_v)
        buu, buv, bvv = _curvature(
            b,
            wp.quat_rotate_inv(rotation_b, normal),
            wp.quat_rotate_inv(rotation_b, tangent_u),
            wp.quat_rotate_inv(rotation_b, tangent_v),
        )
        huu = auu + buu - depth
        huv = auv + buv
        hvv = avv + bvv - depth
        determinant = huu * hvv - huv * huv
        if (
            distance <= bound + accuracy
            and huu >= -accuracy
            and hvv >= -accuracy
            and determinant >= -accuracy * accuracy
        ):
            # An ellipsoid with support height h contains a tangent ball of
            # radius min(radii)^2 / h. For unit u and n, weighted Cauchy-Schwarz
            # gives h(u) >= dot(support(n), u) + R * (1 - dot(n, u)), which is
            # exactly the support inequality for this interior tangent ball.
            # Adding A's ball to the reflected B ball
            # gives an interior ball of their Minkowski difference. Its depth
            # lower-bounds the true minimum, including other local minima.
            height_a = wp.dot(point_a, normal)
            height_b = -wp.dot(point_b - position, normal)
            radius = wp.min(a) * wp.min(a) / height_a + wp.min(b) * wp.min(b) / height_b
            lower_bound = radius - wp.length(delta - radius * normal) - 2e-7 * tolerance
            certified_depth = lower_bound > 0.0 and distance - lower_bound <= accuracy
            if not certified_depth:
                # More iterations in the same stationary basin cannot certify
                # a different, shallower minimum. Keep the general fallback.
                if wp.length(delta - depth * normal) <= accuracy:
                    return False, 0.0, center, normal
            # At shallow depth, normalizing the rounded witness difference
            # amplifies its tangential error. Retain the search normal when
            # midpoint reconstruction moves each surface witness by at most
            # float32 roundoff, as in the general convex raycast refinement.
            if certified_depth and wp.length(delta - depth * normal) <= 2e-7 * tolerance:
                return True, -depth * scale, 0.5 * scale * (point_a + point_b), normal
            candidate = delta / distance
            _, _, certificate = _support_pair(a, b, rotation_b, position, candidate)
            deficit = wp.dot(certificate, candidate) - distance
            if certified_depth and wp.isfinite(distance) and wp.abs(deficit) <= accuracy:
                return True, -distance * scale, 0.5 * scale * (point_a + point_b), candidate

        gu = wp.dot(delta, tangent_u)
        gv = wp.dot(delta, tangent_v)
        step = -(gu * tangent_u + gv * tangent_v) / tolerance
        if huu > 1e-8 and hvv > 1e-8 and determinant > 1e-12:
            step = ((huv * gv - hvv * gu) * tangent_u + (huv * gu - huu * gv) * tangent_v) / determinant
        length = wp.length(step)
        if length > 0.5:
            step *= 0.5 / length
        next_normal = normal
        for _search in range(6):
            next_normal = wp.normalize(normal + step)
            _, _, next_delta = _support_pair(a, b, rotation_b, position, next_normal)
            next_depth = wp.dot(next_delta, next_normal)
            # Near a minimum the decrease in depth is smaller than float32
            # rounding, while the tangential residual can still improve.
            # This changes step selection, never the contact acceptance budget.
            reduced_residual = wp.length_sq(next_delta - next_depth * next_normal) < wp.length_sq(
                delta - depth * normal
            )
            if next_depth < depth or (wp.abs(next_depth - depth) <= 2e-7 * tolerance and reduced_residual):
                break
            step *= 0.5
        normal = next_normal
    return False, 0.0, center, normal
