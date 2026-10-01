# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coulomb friction helpers: the contact tangent basis and the coupled tangent-pair solve.

``FRICTION_PAIR_CUDA`` is a CUDA snippet spliced into the native Gauss-Seidel kernel. It
minimizes the two-tangent quadratic of one contact on the Coulomb disk of the current
normal impulse: sticking uses the block inverse, sliding solves ``(A + alpha I) x = b`` with
``|x| = radius`` by safeguarded Newton with a bisection fallback, and feasible sliding
impulses that already satisfy the KKT conditions to a relative accuracy of 2e-7 are kept
to avoid round-off cycles.
"""

import warp as wp


@wp.func
def contact_tangent_basis(n: wp.vec3):
    """Return the deterministic tangent pair every FeatherPGS friction row uses for ``n``."""
    tangent0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
    if wp.length_sq(tangent0) < 1.0e-12:
        tangent0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    tangent0 = wp.normalize(tangent0)
    tangent1 = wp.normalize(wp.cross(n, tangent0))
    return tangent0, tangent1


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
