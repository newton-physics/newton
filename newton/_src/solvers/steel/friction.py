# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coulomb friction helpers: the contact tangent basis and the coupled tangent-pair solve.

:func:`friction_pair_candidate` (Warp, used by the CPU kernels) and ``FRICTION_PAIR_CUDA`` (a
CUDA snippet spliced into the native Gauss-Seidel kernels) implement the same solve. It
minimizes the two-tangent quadratic of one contact on the Coulomb disk of the current
normal impulse: sticking uses the block inverse, sliding solves ``(A + alpha I) x = b`` with
``|x| = radius`` by safeguarded Newton with a bisection fallback, and feasible sliding
impulses that already satisfy the KKT conditions to a relative accuracy of 2e-7 are kept
to avoid round-off cycles.
"""

import warp as wp


@wp.func
def contact_tangent_basis(n: wp.vec3):
    """Return the deterministic tangent pair every Steel friction row uses for ``n``."""
    tangent0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
    if wp.length_sq(tangent0) < 1.0e-12:
        tangent0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    tangent0 = wp.normalize(tangent0)
    tangent1 = wp.normalize(wp.cross(n, tangent0))
    return tangent0, tangent1


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


_COUPLED_CONTACT_HEAD = """
    // One contact's normal and sliding pair solved together on its 3x3 Delassus block G, from the
    // velocities u (impulses old included), the normal target velocity, and the coefficient mu.
    const auto contact_block_solve = [&](
        float u_n, float u_t1, float u_t2,
        float G_nn, float G_nt1, float G_nt2, float G_t1t1, float G_t1t2, float G_t2t2,
        float target_vel_n, float mu, float old_lambda_n, float old_lambda_t1, float old_lambda_t2,
        float& new_lambda_n, float& new_lambda_t1, float& new_lambda_t2) {
        new_lambda_n = old_lambda_n;
        new_lambda_t1 = old_lambda_t1;
        new_lambda_t2 = old_lambda_t2;
        // A singular block (a link with fewer than two tangential mobilities) keeps the normal
        // impulse its contact row just solved and takes the fixed_normal pair on that disk.
        // Rounding leaves a locked tangent a tiny response, so the test is relative to the trace.
        const auto fixed_normal_pair = [&]() {
            const float radius = fmaxf(mu * old_lambda_n, 0.0f);
            const float2 pair = friction_pair_candidate(
                G_t1t1, G_t1t2, G_t2t2, u_t1, u_t2, old_lambda_t1, old_lambda_t2, radius, 1.0f);
            const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
            const float scale = mag > radius ? radius / mag : 1.0f;
            new_lambda_t1 = pair.x * scale;
            new_lambda_t2 = pair.y * scale;
        };
        const float tangent_trace = G_t1t1 + G_t2t2;
        if (G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2 <= 1.0e-6f * tangent_trace * tangent_trace) {
            fixed_normal_pair();
            return;
        }
"""

# Repeats the arithmetic of _MF_BISECTION_FRICTION_SOURCE in solver_steel.py; change both together.
_COUPLED_CONTACT_BISECTION = """
        if (G_nn < 1.0e-20f) return;
        // At zero load the cone also zeroes the sliding pair.
        float u_n_at_zero = u_n - G_nn * old_lambda_n - G_nt1 * old_lambda_t1 - G_nt2 * old_lambda_t2;
        if (u_n_at_zero >= target_vel_n) {
            new_lambda_n = 0.0f;
            new_lambda_t1 = 0.0f;
            new_lambda_t2 = 0.0f;
            return;
        }
        float lo = 0.0f;
        float hi = fmaxf(old_lambda_n * 2.0f, (target_vel_n - u_n) / G_nn + old_lambda_n);
        hi = fmaxf(hi, 1.0f);
        float f_lo = u_n_at_zero - target_vel_n, f_hi = -1.0f;
        for (int _bi = 0; _bi < 20; _bi++) {
            float mid = 0.5f * (lo + hi);
            float d_n = mid - old_lambda_n;
            float ut1_eff = u_t1 + G_nt1 * d_n;
            float ut2_eff = u_t2 + G_nt2 * d_n;
            float det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
            float d_t1 = 0.0f, d_t2 = 0.0f;
            if (fabsf(det) > 1.0e-20f) {
                d_t1 = (-ut1_eff * G_t2t2 + ut2_eff * G_t1t2) / det;
                d_t2 = ( ut1_eff * G_t1t2 - ut2_eff * G_t1t1) / det;
            }
            float trial_t1 = old_lambda_t1 + d_t1;
            float trial_t2 = old_lambda_t2 + d_t2;
            float flimit = mu * mid;
            float tmag = sqrtf(trial_t1 * trial_t1 + trial_t2 * trial_t2);
            if (tmag > flimit && tmag > 1.0e-20f) {
                float sc = flimit / tmag;
                trial_t1 *= sc;
                trial_t2 *= sc;
            }
            float d_t1_actual = trial_t1 - old_lambda_t1;
            float d_t2_actual = trial_t2 - old_lambda_t2;
            float u_n_trial = u_n + G_nn * d_n + G_nt1 * d_t1_actual + G_nt2 * d_t2_actual;
            if (u_n_trial < target_vel_n) { lo = mid; f_lo = u_n_trial - target_vel_n; }
            else { hi = mid; f_hi = u_n_trial - target_vel_n; }
        }
        // Interpolate the root in the final bracket, whose width scales with the initial one.
        new_lambda_n = f_hi >= 0.0f ? lo - f_lo * (hi - lo) / (f_hi - f_lo) : 0.5f * (lo + hi);
        float d_n_final = new_lambda_n - old_lambda_n;
        float ut1_f = u_t1 + G_nt1 * d_n_final;
        float ut2_f = u_t2 + G_nt2 * d_n_final;
        float det_f = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
        float d_t1_f = 0.0f, d_t2_f = 0.0f;
        if (fabsf(det_f) > 1.0e-20f) {
            d_t1_f = (-ut1_f * G_t2t2 + ut2_f * G_t1t2) / det_f;
            d_t2_f = ( ut1_f * G_t1t2 - ut2_f * G_t1t1) / det_f;
        }
        new_lambda_t1 = old_lambda_t1 + d_t1_f;
        new_lambda_t2 = old_lambda_t2 + d_t2_f;
        float flimit_f = mu * new_lambda_n;
        float tmag_f = sqrtf(new_lambda_t1 * new_lambda_t1 + new_lambda_t2 * new_lambda_t2);
        if (tmag_f > flimit_f && tmag_f > 1.0e-20f) {
            float sc_f = flimit_f / tmag_f;
            new_lambda_t1 *= sc_f;
            new_lambda_t2 *= sc_f;
        }
    };
"""

# Repeats the arithmetic of _MF_COULOMB_NEWTON_FRICTION_SOURCE in solver_steel.py; change both together.
_COUPLED_CONTACT_COULOMB_NEWTON = """
        if (G_nn < 1.0e-20f) return;
        float u_free_n = u_n - (G_nn * old_lambda_n + G_nt1 * old_lambda_t1 + G_nt2 * old_lambda_t2);
        float u_free_t1 = u_t1 - (G_nt1 * old_lambda_n + G_t1t1 * old_lambda_t1 + G_t1t2 * old_lambda_t2);
        float u_free_t2 = u_t2 - (G_nt2 * old_lambda_n + G_t1t2 * old_lambda_t1 + G_t2t2 * old_lambda_t2);
        if (u_free_n >= target_vel_n) {
            new_lambda_n = 0.0f;
            new_lambda_t1 = 0.0f;
            new_lambda_t2 = 0.0f;
            return;
        }
        float bN = u_free_n - target_vel_n;
        float bT0 = u_free_t1;
        float bT1 = u_free_t2;
        float WN = G_nn;
        float wNT0 = G_nt1;
        float wNT1 = G_nt2;
        float AT00 = G_t1t1 - (wNT0 * wNT0) / WN;
        float AT01 = G_t1t2 - (wNT0 * wNT1) / WN;
        float AT10 = G_t1t2 - (wNT1 * wNT0) / WN;
        float AT11 = G_t2t2 - (wNT1 * wNT1) / WN;
        if (AT00 * AT11 - AT01 * AT10 <= 1.0e-6f * tangent_trace * tangent_trace) {
            fixed_normal_pair();
            return;
        }
        float cT0 = bT0 - (bN / WN) * wNT0;
        float cT1 = bT1 - (bN / WN) * wNT1;
        float inv_det0 = 1.0f / (AT00 * AT11 - AT01 * AT10);
        float s0_0 = (AT11 * cT0 - AT01 * cT1) * inv_det0;
        float s0_1 = (AT00 * cT1 - AT10 * cT0) * inv_det0;
        float norm_s0 = sqrtf(s0_0 * s0_0 + s0_1 * s0_1);
        float phi0 = norm_s0 - mu * (wNT0 * s0_0 + wNT1 * s0_1 - bN) / WN;
        float last_s0 = 0.0f, last_s1 = 0.0f;
        if (phi0 <= 0.0f) {
            last_s0 = s0_0;
            last_s1 = s0_1;
        } else {
            float hi = 1.0f;
            for (int _e = 0; _e < 30; _e++) {
                float a_hi = AT00 + hi;
                float d_hi = AT11 + hi;
                float idet_hi = 1.0f / (a_hi * d_hi - AT01 * AT10);
                float sh0 = (d_hi * cT0 - AT01 * cT1) * idet_hi;
                float sh1 = (a_hi * cT1 - AT10 * cT0) * idet_hi;
                float ns_hi = sqrtf(sh0 * sh0 + sh1 * sh1);
                float phi_hi = ns_hi - mu * (wNT0 * sh0 + wNT1 * sh1 - bN) / WN;
                if (phi_hi < 0.0f) break;
                hi *= 2.0f;
            }
            float lo = 0.0f;
            float x = 0.5f * (lo + hi);
            float tol = 1.0e-6f + 1.0e-6f * phi0;
            for (int _it = 0; _it < 50; _it++) {
                float ax = AT00 + x;
                float dx = AT11 + x;
                float idet = 1.0f / (ax * dx - AT01 * AT10);
                float s0 = (dx * cT0 - AT01 * cT1) * idet;
                float s1 = (ax * cT1 - AT10 * cT0) * idet;
                float t0 = (dx * s0 - AT01 * s1) * idet;
                float t1 = (ax * s1 - AT10 * s0) * idet;
                float ns = sqrtf(s0 * s0 + s1 * s1);
                float fx = ns - mu * (wNT0 * s0 + wNT1 * s1 - bN) / WN;
                float dfx = -(s0 * t0 + s1 * t1) / ns + mu * (wNT0 * t0 + wNT1 * t1) / WN;
                last_s0 = s0;
                last_s1 = s1;
                if (fabsf(fx) < tol || fabsf(hi - lo) < 1.0e-6f * (1.0f + hi)) break;
                if (fx > 0.0f) lo = x;
                else hi = x;
                float x_new = 0.5f * (lo + hi);
                if (dfx != 0.0f) {
                    float x_newton = x - fx / dfx;
                    if (x_newton > lo && x_newton < hi) x_new = x_newton;
                }
                x = x_new;
            }
        }
        float rT0 = -last_s0;
        float rT1 = -last_s1;
        new_lambda_n = -(wNT0 * rT0 + wNT1 * rT1 + bN) / WN;
        new_lambda_t1 = rT0;
        new_lambda_t2 = rT1;
        // The root search stops within a tolerance and leaves a residual tangent at mu = 0.
        const float flimit = fmaxf(mu * new_lambda_n, 0.0f);
        const float tmag = sqrtf(new_lambda_t1 * new_lambda_t1 + new_lambda_t2 * new_lambda_t2);
        if (tmag > flimit) {
            const float sc = flimit / tmag;
            new_lambda_t1 *= sc;
            new_lambda_t2 *= sc;
        }
    };
"""


def coupled_contact_cuda(friction_mode: str) -> str:
    """CUDA for ``contact_block_solve``, the coupled contact update of a ``friction_mode`` other than fixed_normal.

    The arithmetic is the matrix-free Gauss-Seidel kernel's for its free-rigid contact rows.
    """
    if friction_mode == "coulomb_newton":
        return _COUPLED_CONTACT_HEAD + _COUPLED_CONTACT_COULOMB_NEWTON
    if friction_mode == "bisection_desaxce":
        # de Saxce: the normal target velocity gains mu times the tangential speed.
        bias = "        target_vel_n = target_vel_n + mu * sqrtf(u_t1 * u_t1 + u_t2 * u_t2);\n"
        return _COUPLED_CONTACT_HEAD + bias + _COUPLED_CONTACT_BISECTION
    if friction_mode == "bisection":
        return _COUPLED_CONTACT_HEAD + _COUPLED_CONTACT_BISECTION
    raise ValueError(f"friction_mode {friction_mode!r} has no coupled contact solve")
