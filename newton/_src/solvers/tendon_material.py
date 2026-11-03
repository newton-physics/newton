# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct material transfer for linear, finite-friction tendon components.

The taut return map preserves total effective extension and satisfies the
capstan cone and net-transfer direction at every internal guide. Its cold solve
uses a piecewise-linear derivative recurrence in tension, with double-precision
intermediates; a valid cached face set reduces subsequent solves to linear work.

Slack has no unique elastic-energy minimizer. Preserve an already feasible
allocation; otherwise choose the feasible allocation with least squared net
material transfer. Redistribution supports nonzero trial extensions and slack
bounds spanning at most 2**68 in magnitude.

Upper extension bounds are feasibility checks, not additional contact forces.
An incompatible bound produces an explicit failure, never a clipped result or a
fallback to sweeps. The caller must invalidate cached faces if route identities
change and must handle negative status before accepting a simulation result.
"""

from enum import IntEnum

import warp as wp


class TendonMaterialStatus(IntEnum):
    """Result of a direct tendon material-component solve."""

    COLD = 1
    """Solve successfully using the direct derivative recurrence."""

    WARM = 2
    """Solve successfully using the cached active faces."""

    SLACK_UNCHANGED = 3
    """Preserve a feasible zero-tension material allocation."""

    SLACK_REDISTRIBUTED = 4
    """Redistribute slack with the least squared net material transfer."""

    BAD_INPUT = -1
    """Reject unsupported dimensions or nonfinite/out-of-range inputs."""

    SLACK_INFEASIBLE = -2
    """Reject bounds incompatible with zero tension and conserved material."""

    BOUND_INCOMPATIBLE = -3
    """Reject a taut return map that exceeds an effective-extension bound."""

    NUMERICAL_FAILURE = -4
    """Reject a result that cannot pass the numerical consistency checks."""

    INVALID_ROUTE = -5
    """Reject inconsistent active-span connections."""

    COMPONENT_TOO_LARGE = -6
    """Reject a material component with more than 32 active spans."""


@wp.func_native("""
const int offset = tendon * span_stride;
const int edge_offset = tendon * edge_stride;
status[tendon] = -1;
piece_count[tendon] = 0;
if (n < 1 || n > 32) return;
double total = 0.0, total_error = 0.0;
for (int i = 0; i < n; ++i) {
    const double x = initial[offset + i];
    const double c = compliance[offset + i];
    const double u = upper[offset + i];
    if (!wp::isfinite(x) || !wp::isfinite(c) || c < 1.0e-25 ||
        !wp::isfinite(u)) return;
    const double next_total = total + x;
    total_error += wp::abs(total) >= wp::abs(x) ? (total - next_total) + x : (x - next_total) + total;
    total = next_total;
    if (i < n - 1) {
        const double k = cap_ratio[edge_offset + i];
        if (!wp::isfinite(k) || k < 1.0 || k > 4.0) return;
    }
}
total += total_error;
if (total <= 0.0) {
    bool unchanged = true;
    double required = 0.0, required_error = 0.0;
    double minimum_input = 1.0e300, maximum_input = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = wp::min(double(upper[offset + i]), 0.0);
        for (int part = 0; part < 2; ++part) {
            const double value = part == 0 ? double(initial[offset + i]) : -v;
            if (value != 0.0) minimum_input = wp::min(minimum_input, wp::abs(value));
            maximum_input = wp::max(maximum_input, wp::abs(value));
            const double next = required + value;
            required_error += wp::abs(required) >= wp::abs(value) ? (required - next) + value : (value - next) + required;
            required = next;
        }
        if (initial[offset + i] > v) unchanged = false;
    }
    double result[32];
    if (unchanged) {
        for (int i = 0; i < n; ++i) result[i] = initial[offset + i];
    } else {
        // Two-component arithmetic is not an arbitrary-precision accumulator.
        // Preserve valid histories above, but reject redistribution beyond the compensated accumulator's supported range.
        if (maximum_input > 295147905179352825856.0 * minimum_input) { status[tendon] = -4; return; }
        // Sum x0-V directly: subtracting two large totals can hide a real bound deficit.
        const double capacity = -required, capacity_error = -required_error;
        if (capacity + capacity_error < 0.0) { status[tendon] = -2; return; }
        double block_sum[31], block_error[31], fit[32], fit_error[32];
        int block_count[31], block_end[31];
        int blocks = 0;
        double prefix = 0.0, prefix_error = 0.0;
        // PAVA fits -prefix(x0-V) to nondecreasing values in [0, capacity].
        // This minimizes squared net guide transfer, not a physical slack energy.
        for (int i = 0; i < n - 1; ++i) {
            for (int part = 0; part < 2; ++part) {
                const double value = part == 0 ? double(initial[offset + i]) : -wp::min(double(upper[offset + i]), 0.0);
                const double next = prefix + value;
                prefix_error += wp::abs(prefix) >= wp::abs(value) ? (prefix - next) + value : (value - next) + prefix;
                prefix = next;
            }
            // Retain the low part: large opposite prefixes may cancel only after pooling.
            block_sum[blocks] = -prefix;
            block_error[blocks] = -prefix_error;
            block_count[blocks] = 1;
            block_end[blocks] = i + 1;
            ++blocks;
            while (blocks > 1) {
                const int left = blocks - 2, right = blocks - 1;
                const double left_mean = block_sum[left] / block_count[left];
                const double right_mean = block_sum[right] / block_count[right];
                const double left_error = (::fma(-left_mean, double(block_count[left]), block_sum[left]) + block_error[left]) / block_count[left];
                const double right_error = (::fma(-right_mean, double(block_count[right]), block_sum[right]) + block_error[right]) / block_count[right];
                if ((left_mean - right_mean) + (left_error - right_error) <= 0.0) break;
                const double merged = block_sum[left] + block_sum[right];
                block_error[left] += block_error[right] + (wp::abs(block_sum[left]) >= wp::abs(block_sum[right])
                    ? (block_sum[left] - merged) + block_sum[right] : (block_sum[right] - merged) + block_sum[left]);
                block_sum[left] = merged;
                block_count[blocks - 2] += block_count[blocks - 1];
                block_end[blocks - 2] = block_end[blocks - 1];
                --blocks;
            }
        }
        int first = 0;
        for (int b = 0; b < blocks; ++b) {
            const double mean = block_sum[b] / block_count[b];
            const double mean_error = (::fma(-mean, double(block_count[b]), block_sum[b]) + block_error[b]) / block_count[b];
            double value = mean, value_error = mean_error;
            if (mean + mean_error < 0.0) { value = 0.0; value_error = 0.0; }
            else if ((mean - capacity) + (mean_error - capacity_error) > 0.0) {
                value = capacity; value_error = capacity_error;
            }
            for (int i = first; i < block_end[b]; ++i) { fit[i] = value; fit_error[i] = value_error; }
            first = block_end[b];
        }
        fit[n - 1] = capacity;
        fit_error[n - 1] = capacity_error;
        double previous = 0.0, previous_error = 0.0, result_total = 0.0, rounded_total = 0.0, cast_budget = 0.0;
        for (int i = 0; i < n; ++i) {
            const double v = wp::min(double(upper[offset + i]), 0.0);
            // Monotone gaps enforce x<=V exactly, without cancellation in x0+B*z.
            const double delta = fit[i] - previous;
            const double delta_error = (wp::abs(fit[i]) >= wp::abs(previous)
                ? (fit[i] - delta) - previous : (-previous - delta) + fit[i]);
            const double gap = delta + (delta_error + fit_error[i] - previous_error);
            if (gap < 0.0) { status[tendon] = -4; return; }
            result[i] = v - gap;
            previous = fit[i];
            previous_error = fit_error[i];
            const float rounded = static_cast<float>(result[i]);
            if (!wp::isfinite(rounded) || double(rounded) > v) { status[tendon] = -4; return; }
            result_total += result[i];
            rounded_total += rounded;
            cast_budget += wp::abs(double(rounded) - result[i]);
        }
        // All outputs are nonpositive: conservation is relative to actual slack,
        // not to the possibly much larger cancelling positive/negative trial values.
        const double roundoff = 128.0 * n * 2.220446049250313e-16 * wp::max(wp::abs(total), wp::abs(result_total));
        if (wp::abs(result_total - total) > roundoff || wp::abs(rounded_total - total) > roundoff + cast_budget) {
            status[tendon] = -4; return;
        }
    }
    double flow = 0.0, flow_error = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!unchanged) {
            for (int part = 0; part < 2; ++part) {
                const double value = part == 0 ? double(initial[offset + i]) : -result[i];
                const double next = flow + value;
                flow_error += wp::abs(flow) >= wp::abs(value) ? (flow - next) + value : (value - next) + flow;
                flow = next;
            }
        }
        // Both component boundaries have exactly zero material flux.
        transfer[offset + i] = i < n - 1 ? flow + flow_error : 0.0;
        output[offset + i] = static_cast<float>(result[i]);
    }
    for (int i = 0; i < n - 1; ++i) faces[edge_offset + i] = 0;
    cache_valid[tendon] = 0;
    status[tendon] = unchanged ? 3 : 4;
    return;
}


double tension[32];
// One cached-face attempt, then one deterministic direct cold solve.
for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt == 0) {
        if (allow_warm == 0 || cache_valid[tendon] == 0) continue;
        bool have_solution = true;
        int first = 0;
        while (first < n && have_solution) {
            int last = first;
            while (last < n - 1 && faces[edge_offset + last] != 0) ++last;
            double weighted_compliance = 0.0, block_extension = 0.0;
            double weight = 1.0;
            for (int i = first; i <= last; ++i) {
                if (i > first) {
                    const int face = faces[edge_offset + i - 1];
                    if (face != -1 && face != 1) { have_solution = false; break; }
                    const double k = cap_ratio[edge_offset + i - 1];
                    weight = face > 0 ? weight / k : weight * k;
                }
                tension[i] = weight;
                weighted_compliance += compliance[offset + i] * weight;
                block_extension += initial[offset + i];
            }
            if (!have_solution || !(weighted_compliance > 0.0) || !(block_extension > 0.0)) {
                have_solution = false; break;
            }
            const double base = block_extension / weighted_compliance;
            for (int i = first; i <= last; ++i) tension[i] *= base;
            first = last + 1;
        }

        if (!have_solution) continue;
    } else {
        double low[2][63], high[2][63], aa[2][63], bb[2][63];
        double minima[32];
        int current = 0;
        int count = 1;
        int maximum_count = 1;
        low[0][0] = 0.0;
        high[0][0] = 1.0e30;
        aa[0][0] = compliance[offset];
        bb[0][0] = initial[offset];
        minima[0] = bb[0][0] > 0.0 ? bb[0][0] / aa[0][0] : 0.0;

        for (int i = 1; i < n; ++i) {
            const int next = 1 - current;
            const double m = minima[i - 1];
            const double k = cap_ratio[edge_offset + i - 1];
            const double c = compliance[offset + i];
            const double x = initial[offset + i];
            int used = 0;
            if (m > 0.0) {
                for (int j = 0; j < count; ++j) {
                    if (low[current][j] < m) {
                        if (used >= 63) { status[tendon] = -4; return; }
                        low[next][used] = low[current][j] / k;
                        high[next][used] = wp::min(high[current][j], m) / k;
                        aa[next][used] = aa[current][j] * k + c;
                        bb[next][used] = bb[current][j] + x;
                        ++used;
                    }
                }
                if (k > 1.0) {
                    if (used >= 63) { status[tendon] = -4; return; }
                    low[next][used] = m / k;
                    high[next][used] = m * k;
                    aa[next][used] = c;
                    bb[next][used] = x;
                    ++used;
                }
            }
            for (int j = 0; j < count; ++j) {
                if (high[current][j] > m) {
                    if (used >= 63) { status[tendon] = -4; return; }
                    low[next][used] = wp::max(low[current][j], m) * k;
                    high[next][used] = wp::min(high[current][j] * k, 1.0e30);
                    aa[next][used] = aa[current][j] / k + c;
                    bb[next][used] = bb[current][j] + x;
                    ++used;
                }
            }
            if (used == 0) { status[tendon] = -4; return; }
            count = used;
            current = next;
            maximum_count = wp::max(maximum_count, count);
            double root = 0.0;
            if (bb[current][0] > 0.0) {
                bool found = false;
                for (int j = 0; j < count; ++j) {
                    if (!(aa[current][j] > 0.0) || !wp::isfinite(aa[current][j]) ||
                        !wp::isfinite(bb[current][j])) { status[tendon] = -4; return; }
                    if (bb[current][j] > 0.0) {
                        const double candidate = bb[current][j] / aa[current][j];
                        if (candidate >= low[current][j] && candidate <= high[current][j]) {
                            root = candidate;
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) { status[tendon] = -4; return; }
            }
            minima[i] = root;
        }

        tension[n - 1] = minima[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            const double k = cap_ratio[edge_offset + i];
            tension[i] = wp::max(tension[i + 1] / k, wp::min(tension[i + 1] * k, minima[i]));
        }

        piece_count[tendon] = maximum_count;
    }
    int validation_status = 1;
    double edge_transfer[32];
    double flow = 0.0, compensation = 0.0;
    double minimum_compliance = 1.0e300, peak_tension = 0.0;
    for (int i = 0; i < n; ++i) {
        const double c = compliance[offset + i];
        const double value = c * tension[i];
        if (!wp::isfinite(value) || !(tension[i] > 0.0)) { validation_status = -4; break; }
        // Bound contact needs a defined reaction model; do not validate virtual stress.
        const double upper_value = upper[offset + i];
        const double bound_roundoff = 8.0 * 2.220446049250313e-16 * wp::max(wp::abs(value), wp::abs(upper_value));
        if (value > upper_value + bound_roundoff) { validation_status = -3; break; }
        minimum_compliance = wp::min(minimum_compliance, c);
        peak_tension = wp::max(peak_tension, tension[i]);
        const double residual = ::fma(-c, tension[i], double(initial[offset + i])) - compensation;
        const double next_flow = flow + residual;
        compensation = (next_flow - flow) - residual;
        flow = next_flow;
        edge_transfer[i] = i < n - 1 ? flow : 0.0;
        if (i < n - 1) {
            const double left = tension[i], right = tension[i + 1];
            const double k = cap_ratio[edge_offset + i];
            const double force_tolerance = 2.0e-6 * wp::max(wp::max(left, right), 1.0e-20);
            // Do not let a user-sized force tolerance merge both finite-friction faces.
            const double face_roundoff = 64.0 * double(n) * 2.220446049250313e-16 * wp::max(wp::max(left, right), 1.0e-20);
            const double c_next = compliance[offset + i + 1];
            const double flow_tolerance = force_tolerance / wp::max(1.0/c + k/c_next, k/c + 1.0/c_next);
            const double positive_gap = k * right - left;
            const double negative_gap = k * left - right;
            bool valid = true;
            if (positive_gap < -face_roundoff || negative_gap < -face_roundoff) valid = false;
            else if (positive_gap > face_roundoff && negative_gap > face_roundoff) {
                valid = wp::abs(flow) <= flow_tolerance;
                edge_transfer[i] = 0.0;
            }
            else if (positive_gap <= face_roundoff && negative_gap > face_roundoff)
                valid = flow >= -flow_tolerance;
            else if (negative_gap <= face_roundoff && positive_gap > face_roundoff)
                valid = flow <= flow_tolerance;
            if (!valid) { validation_status = -4; break; }
        }
    }
    if (validation_status == 1 && wp::abs(flow) > 2.0e-6 * wp::max(peak_tension, 1.0e-20) * minimum_compliance) {
        validation_status = -4;
    }
    for (int i = 0; i < n && validation_status == 1; ++i) {
        // Float32 storage must remain finite and respect the rest bound too.
        const float rounded = static_cast<float>(compliance[offset + i] * tension[i]);
        if (!wp::isfinite(rounded) || !(rounded > 0.0f) || rounded > upper[offset + i] ||
            wp::abs(double(rounded) / compliance[offset + i] - tension[i]) > 2.0e-7 * wp::max(tension[i], 1.0e-20)) {
            validation_status = -4; break;
        }
    }

    if (validation_status != 1) {
        if (attempt == 0) continue;
        status[tendon] = validation_status;
        return;
    }
    for (int i = 0; i < n; ++i) {
        output[offset + i] = static_cast<float>(compliance[offset + i] * tension[i]);
        // Keep flux before rounding the damped extension: reconstructing it
        // from output can turn damping roundoff into permanent material drift.
        transfer[offset + i] = edge_transfer[i];
    }
    for (int i = 0; i < n - 1; ++i) {
        const double k = cap_ratio[edge_offset + i];
        const double left = tension[i], right = tension[i + 1];
        const double tolerance = 64.0 * double(n) * 2.220446049250313e-16 * wp::max(wp::max(left, right), 1.0e-20);
        if (wp::abs(left - k * right) <= tolerance) faces[edge_offset + i] = 1;
        else if (wp::abs(right - k * left) <= tolerance) faces[edge_offset + i] = -1;
        else faces[edge_offset + i] = 0;
    }
    cache_valid[tendon] = 1;
    status[tendon] = attempt == 0 ? 2 : 1;
    return;
}
""")
def solve_tendon_material_component(
    initial: wp.array[float],
    compliance: wp.array[float],
    cap_ratio: wp.array[float],
    upper: wp.array[float],
    output: wp.array[float],
    transfer: wp.array[wp.float64],
    status: wp.array[int],
    faces: wp.array[int],
    cache_valid: wp.array[int],
    piece_count: wp.array[int],
    n: int,
    allow_warm: int,
    tendon: int,
    span_stride: int,
    edge_stride: int,
):
    """Project a linear finite-friction component without moving its bodies.

    A cached active-face set is validated first. If it is stale, the
    deterministic direct solve recomputes the same return map. Neither path
    uses material sweeps. Negative status leaves output, transfer, and cached faces
    unchanged; the caller must report the failure instead of accepting it.

    Args:
        initial: Effective trial extensions, including frozen damping [m].
        compliance: Positive frozen segment compliances [m/N], at least 1e-25.
        cap_ratio: Capstan ratios at internal guides, in [1, 4].
        upper: Maximum effective extensions allowed by minimum rest lengths [m].
        output: Projected effective extensions [m].
        transfer: Net material transfer after each span [m]; zero at the component end.
            Double-precision scratch preserves small differences between adjacent
            transfers until the caller updates its float32 material history.
        status: Per-component result code from :class:`TendonMaterialStatus`.
        faces: Cached signed capstan faces, -1, 0, or 1.
        cache_valid: Whether the cached faces belong to the current component.
        piece_count: Number of derivative pieces visited by the cold solve.
        n: Number of active spans, from 1 to 32.
        allow_warm: Whether to try a valid cached face set.
        tendon: Component index for status and cache metadata.
        span_stride: Multiplier of the component index for span array offsets.
        edge_stride: Multiplier of the component index for guide array offsets.
    """
    ...
