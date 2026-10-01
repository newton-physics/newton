# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve the tangential friction block at a fixed normal load."""

import warp as wp


@wp.func
def friction_pair_candidate(
    a: float, c: float, d: float, residual: wp.vec2, old: wp.vec2, radius: float, omega: float
) -> wp.vec2:
    """Minimize the tangent quadratic on a disk, then apply relaxation.

    The caller projects the relaxed result back onto the disk. Sticking uses
    the block inverse; sliding solves (A + alpha I)x = b with |x| = radius.
    Incremental sticking avoids perturbing fixed points. An orthonormal
    eigenbasis avoids cancellation in nearly singular blocks.
    Safeguarded Newton accelerates sliding; bisection retains a feasible
    fallback. Feasible sliding impulses already satisfying the KKT conditions
    to the same relative accuracy (2e-7) are retained to avoid roundoff cycles.
    Normal load is fixed in this solve.
    """
    if radius <= 0.0:
        return wp.vec2(0.0)
    scale = wp.max(wp.max(a, d), 1.0e-20)
    a /= scale
    c /= scale
    d /= scale
    largest = 0.5 * (a + d + wp.sqrt((a - d) * (a - d) + 4.0 * c * c))
    smallest = float(0.0)
    if largest > 0.0:
        smallest = wp.max((a * d - c * c) / largest, 0.0)
    axis = wp.vec2(c, largest - a)
    if a >= d:
        axis = wp.vec2(largest - d, c)
    if wp.length(axis) > 0.0:
        axis = wp.normalize(axis)
    else:
        axis = wp.vec2(1.0, 0.0)
    perpendicular = wp.vec2(-axis[1], axis[0])
    residual_rotated = wp.vec2(wp.dot(axis, residual), wp.dot(perpendicular, residual)) / scale
    result = old
    sticking = bool(False)
    if largest > 0.0 and (smallest > 0.0 or residual_rotated[1] == 0.0):
        correction = (residual_rotated[0] / largest) * axis
        if smallest > 0.0:
            correction += (residual_rotated[1] / smallest) * perpendicular
        # Incremental sticking preserves a feasible zero-residual impulse exactly.
        result = old - correction
        sticking = wp.length(result) <= radius
    if not sticking:
        # Retain an already feasible sliding solution within the same relative
        # accuracy as the root solve. Re-solving it can otherwise introduce
        # roundoff cycles that prevent a stationary PGS sweep.
        old_norm = wp.length(old)
        residual_norm = wp.length(residual)
        cross_residual = residual[0] * old[1] - residual[1] * old[0]
        if (
            old_norm > 0.0
            and old_norm <= radius
            and radius - old_norm <= 2.0e-7 * radius
            and wp.dot(residual, old) <= 0.0
            and wp.abs(cross_residual) <= 2.0e-7 * old_norm * residual_norm
        ):
            return old
        old_rotated = wp.vec2(wp.dot(axis, old), wp.dot(perpendicular, old))
        b = wp.vec2(largest * old_rotated[0], smallest * old_rotated[1]) - residual_rotated
        solution = wp.vec2(0.0)
        lo = float(0.0)
        hi = wp.length(b) / radius
        if hi > 0.0:
            # Both bounds are <= the multiplier root; together they also bound
            # the starting norm by sqrt(2) * radius. The reciprocal norm is
            # increasing and concave, giving Newton a useful step near singularity.
            lo = wp.max(wp.max(hi - largest, wp.abs(b[1]) / radius - smallest), 0.0)
            alpha = lo
            converged = bool(False)
            for _ in range(8):
                inv0 = 1.0 / (largest + alpha)
                inv1 = float(0.0)
                if smallest + alpha > 0.0:
                    inv1 = 1.0 / (smallest + alpha)
                trial = old_rotated - wp.vec2(
                    (residual_rotated[0] + alpha * old_rotated[0]) * inv0,
                    (residual_rotated[1] + alpha * old_rotated[1]) * inv1,
                )
                norm = wp.length(trial)
                # The stopping error is relative to the impulse budget. The
                # caller projects after relaxation; keep bisection for hard cases.
                if wp.abs(norm - radius) <= 2.0e-7 * radius or (alpha == 0.0 and norm <= radius):
                    solution = trial
                    converged = True
                    break
                if norm > radius:
                    lo = alpha
                else:
                    hi = alpha
                slope = trial[0] * trial[0] * inv0 + trial[1] * trial[1] * inv1
                candidate = alpha
                if slope > 0.0:
                    candidate = alpha + (norm / radius - 1.0) * norm * norm / slope
                alpha = 0.5 * (lo + hi)
                if candidate > lo and candidate < hi:
                    alpha = candidate
            if not converged:
                for _ in range(24):
                    alpha = 0.5 * (lo + hi)
                    trial = wp.vec2(b[0] / (largest + alpha), b[1] / (smallest + alpha))
                    if wp.length(trial) > radius:
                        lo = alpha
                    else:
                        hi = alpha
                solution = wp.vec2(b[0] / (largest + hi), b[1] / (smallest + hi))
        result = old + (solution[0] - old_rotated[0]) * axis + (solution[1] - old_rotated[1]) * perpendicular
    return old + omega * (result - old)


# Native kernels use the same bounded solve as the Warp function above.
FRICTION_PAIR_CUDA = """
    // The pair's 2x2 operator [a c; c d] in its scaled eigenbasis. It depends only on the
    // rows, so a solve that sees the same pair every sweep can compute it once.
    const auto friction_pair_setup = [](
        float a, float c, float d,
        float& scale, float& largest, float& smallest, float& vx, float& vy) {
        scale = fmaxf(fmaxf(a, d), 1.0e-20f);
        a /= scale; c /= scale; d /= scale;
        largest = 0.5f * (a + d + sqrtf((a-d)*(a-d) + 4.0f*c*c));
        smallest = largest > 0.0f ? fmaxf((a*d-c*c) / largest, 0.0f) : 0.0f;
        vx = c; vy = largest-a;
        if (a >= d) { vx = largest-d; vy = c; }
        float norm = sqrtf(vx*vx + vy*vy);
        if (norm > 0.0f) { vx /= norm; vy /= norm; }
        else { vx = 1.0f; vy = 0.0f; }
    };
    const auto friction_pair_solve = [](
        float scale, float largest, float smallest, float vx, float vy,
        float r0, float r1, float old0, float old1, float radius, float omega) {
        if (radius <= 0.0f) return make_float2(0.0f, 0.0f);
        float r0_rotated = (vx*r0 + vy*r1) / scale;
        float r1_rotated = (-vy*r0 + vx*r1) / scale;
        float result0 = old0, result1 = old1;
        bool sticking = false;
        if (largest > 0.0f && (smallest > 0.0f || r1_rotated == 0.0f)) {
            float dx = r0_rotated / largest;
            float dy = smallest > 0.0f ? r1_rotated / smallest : 0.0f;
            result0 = old0 - (vx*dx - vy*dy);
            result1 = old1 - (vy*dx + vx*dy);
            sticking = sqrtf(result0*result0 + result1*result1) <= radius;
        }
        if (!sticking) {
            float old_norm = sqrtf(old0*old0 + old1*old1);
            float residual_norm = sqrtf(r0*r0 + r1*r1);
            if (old_norm > 0.0f && old_norm <= radius
                && radius-old_norm <= 2.0e-7f*radius
                && r0*old0+r1*old1 <= 0.0f
                && fabsf(r0*old1-r1*old0) <= 2.0e-7f*old_norm*residual_norm) {
                return make_float2(old0, old1);
            }
            float old_x = vx*old0 + vy*old1;
            float old_y = -vy*old0 + vx*old1;
            float b0 = largest * old_x - r0_rotated;
            float b1 = smallest * old_y - r1_rotated;
            float x = 0.0f, y = 0.0f;
            float lo = 0.0f;
            float hi = sqrtf(b0*b0 + b1*b1) / radius;
            if (hi > 0.0f) {
                lo = fmaxf(fmaxf(hi-largest, fabsf(b1)/radius-smallest), 0.0f);
                float alpha = lo;
                bool converged = false;
                for (int iteration = 0; iteration < 8; ++iteration) {
                    float inv0 = 1.0f / (largest+alpha);
                    float inv1 = smallest+alpha > 0.0f ? 1.0f / (smallest+alpha) : 0.0f;
                    float tx = old_x - (r0_rotated + alpha*old_x)*inv0;
                    float ty = old_y - (r1_rotated + alpha*old_y)*inv1;
                    float norm = sqrtf(tx*tx + ty*ty);
                    if (fabsf(norm-radius) <= 2.0e-7f * radius || (alpha == 0.0f && norm <= radius)) {
                        x = tx; y = ty;
                        converged = true;
                        break;
                    }
                    if (norm > radius) lo = alpha;
                    else hi = alpha;
                    float slope = tx*tx*inv0 + ty*ty*inv1;
                    float candidate = alpha;
                    if (slope > 0.0f) candidate = alpha + (norm/radius-1.0f)*norm*norm/slope;
                    alpha = 0.5f * (lo+hi);
                    if (candidate > lo && candidate < hi) alpha = candidate;
                }
                if (!converged) {
                    for (int iteration = 0; iteration < 24; ++iteration) {
                        alpha = 0.5f * (lo + hi);
                        float tx = b0 / (largest + alpha);
                        float ty = b1 / (smallest + alpha);
                        if (sqrtf(tx*tx + ty*ty) > radius) lo = alpha;
                        else hi = alpha;
                    }
                    x = b0 / (largest + hi);
                    y = b1 / (smallest + hi);
                }
            }
            result0 = old0 + vx*(x-old_x) - vy*(y-old_y);
            result1 = old1 + vy*(x-old_x) + vx*(y-old_y);
        }
        return make_float2(old0 + omega*(result0-old0), old1 + omega*(result1-old1));
    };
    const auto friction_pair_candidate = [&](
        float a, float c, float d, float r0, float r1,
        float old0, float old1, float radius, float omega) {
        if (radius <= 0.0f) return make_float2(0.0f, 0.0f);
        float scale, largest, smallest, vx, vy;
        friction_pair_setup(a, c, d, scale, largest, smallest, vx, vy);
        return friction_pair_solve(scale, largest, smallest, vx, vy, r0, r1, old0, old1, radius, omega);
    };
"""
