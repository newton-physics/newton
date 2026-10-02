# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-owned friction regions, independent of collision contact identities.

Normal contacts are never reduced here. Each compatible region selects up to two
friction locations, with an equal share of the region's total normal impulse.
Locations lie on the footprint's principal axis through its centroid, at the
members' axial extremes, avoiding a diagonal friction couple on symmetric narrow
footprints. Isotropic footprints retain a canonical farthest pair because they
have no unique axis.
Regions require matching friction materials, nearly aligned normals, nearby
contact planes, and connected shape bounding spheres on each body. The latter
is a conservative test across convex seams, not an exact surface-connectivity
query. Length tolerances scale with the smaller body's bounding radius; angular
tolerance is about six degrees. Normals remain individual unilateral rows.

Tangential displacement and impulses persist across frames; friction locations
follow the current contact region. Displacement accumulates incremental rigid
motion at those locations, using the pose increment's SE(3) logarithm rather
than a rotating material point's chord. Saved poses also capture externally
imposed motion and position-only solver passes. Pair identity and geometric
compatibility correlate history without a collision matcher or shape-type rules.
Unloaded regions and saturated friction with opposing slip release history;
saturation alone does not. All matching tolerances are internal so correction
strength is the only new user control.

Sorting body pairs makes construction local to a pair and preserves CUDA graph
capture: all buffers and the radix-sort workspace have fixed capacity.
"""

import inspect

import numpy as np
import warp as wp

from ...math.spatial import velocity_at_point
from .contact_filters import contact_friction_eligible, contact_normal_gap_limit


@wp.struct
class FrictionPatches:
    """Per-contact view consumed by the existing contact row builders."""

    enabled: int
    weight: wp.array[float]
    next_contact: wp.array[int]
    point_a: wp.array[wp.vec3]
    point_b: wp.array[wp.vec3]
    phi: wp.array[wp.vec2]


@wp.struct
class _PatchFrame:
    keys: wp.array[wp.int64]
    indices: wp.array[int]
    center: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    displacement: wp.array[wp.vec3]
    mu: wp.array[wp.vec2]
    radius: wp.array[float]
    body_a: wp.array[int]
    body_b: wp.array[int]
    shape_a: wp.array[int]
    shape_b: wp.array[int]
    flipped: wp.array[int]
    owner: wp.array[int]
    anchor_a: wp.array[wp.vec3]
    anchor_b: wp.array[wp.vec3]
    surface_a: wp.array[wp.vec3]
    surface_b: wp.array[wp.vec3]
    valid: wp.array[int]
    used: wp.array[int]
    source: wp.array[int]
    eligible: wp.array[int]
    # Scratch indices: patch members in the current frame, valid anchors in the previous frame.
    members: wp.array[int]
    # Per region, by seed (written by the device flood pass): first friction location,
    # end of its member list, the member farthest from the first and that squared distance.
    first: wp.array[int]
    region_stop: wp.array[int]
    second: wp.array[int]
    separation: wp.array[float]
    support_gap_limit: wp.array[float]
    tangent_impulse: wp.array[wp.vec3]


@wp.func
def _world_point(q: wp.array[wp.transform], body: int, point: wp.vec3):
    result = point
    if body >= 0:
        result = wp.transform_point(q[body], point)
    return result


@wp.func
def _local_point(q: wp.array[wp.transform], body: int, point: wp.vec3):
    result = point
    if body >= 0:
        result = wp.transform_point(wp.transform_inverse(q[body]), point)
    return result


@wp.kernel(enable_backward=False)
def _prepare(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    shape_mu: wp.array[float],
    shape_gap: wp.array[float],
    body_radius: wp.array[float],
    q: wp.array[wp.transform],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    contact_gap_gate: float,
    same_articulation_gap_gate: float,
    articulation_pair_gap_gate: float,
    friction_gap: float,
    friction_articulation_pairs_only: int,
    frame: _PatchFrame,
    patches: FrictionPatches,
):
    c = wp.tid()
    frame.indices[c] = c
    frame.keys[c] = wp.int64(0x7FFFFFFFFFFFFFFF)
    frame.owner[c] = -1
    frame.valid[c] = 0
    frame.used[c] = 0
    frame.source[c] = -1
    frame.eligible[c] = 0
    frame.tangent_impulse[c] = wp.vec3(0.0)
    patches.weight[c] = 0.0
    patches.next_contact[c] = -1
    patches.phi[c] = wp.vec2(0.0)
    if count[0] > shape0.shape[0] or c >= count[0] or c >= shape0.shape[0]:
        return
    sa = shape0[c]
    sb = shape1[c]
    a = int(-1)
    b = int(-1)
    if sa >= 0:
        a = shape_body[sa]
    if sb >= 0:
        b = shape_body[sb]
    ka = a + 1
    kb = b + 1
    if a < 0:
        ka = q.shape[0] + (sa if sa >= 0 else shape_body.shape[0]) + 1
    if b < 0:
        kb = q.shape[0] + (sb if sb >= 0 else shape_body.shape[0]) + 1
    n = -normal[c]
    pa = _world_point(q, a, point0[c]) - margin0[c] * n
    pb = _world_point(q, b, point1[c]) + margin1[c] * n
    gap = wp.dot(n, pa - pb)
    art_a = int(-1)
    art_b = int(-1)
    if a >= 0 and a < body_to_articulation.shape[0]:
        art_a = body_to_articulation[a]
    if b >= 0 and b < body_to_articulation.shape[0]:
        art_b = body_to_articulation[b]
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    support_gap_limit = float(0.0)
    if sa >= 0:
        support_gap_limit += shape_gap[sa]
    if sb >= 0:
        support_gap_limit += shape_gap[sb]
    normal_gap_limit = contact_normal_gap_limit(
        a_non_free,
        b_non_free,
        art_a == art_b,
        contact_gap_gate,
        articulation_pair_gap_gate,
        same_articulation_gap_gate,
    )
    support_gap_limit = wp.min(support_gap_limit, normal_gap_limit)
    eligible = gap <= normal_gap_limit and contact_friction_eligible(
        gap, a_non_free, b_non_free, friction_articulation_pairs_only, friction_gap
    )
    # Friction-only filters suppress rows without ending contact support;
    # filtered regions must still be able to carry their anchor history.
    frame.support_gap_limit[c] = wp.max(support_gap_limit, 0.0)
    frame.eligible[c] = int(eligible)
    flip = int(ka > kb)
    if flip != 0:
        t = a
        a = b
        b = t
        t = ka
        ka = kb
        kb = t
        t = sa
        sa = sb
        sb = t
        n = -n
    frame.keys[c] = (wp.int64(ka) << wp.int64(32)) | wp.int64(kb)
    frame.body_a[c] = a
    frame.body_b[c] = b
    frame.shape_a[c] = sa
    frame.shape_b[c] = sb
    frame.flipped[c] = flip
    frame.surface_a[c] = _local_point(q, a, pb if flip != 0 else pa)
    frame.surface_b[c] = _local_point(q, b, pa if flip != 0 else pb)
    frame.center[c] = 0.5 * (pa + pb)
    frame.normal[c] = n
    mu_a = float(0.0)
    mu_b = float(0.0)
    if sa >= 0:
        mu_a = shape_mu[sa]
    if sb >= 0:
        mu_b = shape_mu[sb]
    frame.mu[c] = wp.vec2(mu_a, mu_b)
    r = float(1.0e10)
    if a >= 0:
        r = wp.min(r, body_radius[a])
    if b >= 0:
        r = wp.min(r, body_radius[b])
    frame.radius[c] = wp.max(r, 1.0e-4)


@wp.func
def _compatible(frame: _PatchFrame, seed: int, c: int):
    d = frame.center[c] - frame.center[seed]
    r = frame.radius[seed]
    return (
        wp.dot(frame.normal[seed], frame.normal[c]) >= 0.995
        and frame.mu[seed][0] == frame.mu[c][0]
        and frame.mu[seed][1] == frame.mu[c][1]
        and wp.abs(wp.dot(d, frame.normal[seed])) <= 0.02 * r
        and wp.length_sq(d) <= 4.0 * r * r
    )


@wp.func
def _shapes_adjacent(a: int, b: int, transforms: wp.array[wp.transform], radii: wp.array[float]):
    if a == b:
        return True
    if a < 0 or b < 0:
        return False
    radius = radii[a] + radii[b]
    delta = wp.transform_get_translation(transforms[a]) - wp.transform_get_translation(transforms[b])
    return wp.length_sq(delta) <= (1.0 + 1.0e-5) * radius * radius


@wp.func
def _geometry_adjacent(sa: int, sb: int, ta: int, tb: int, transforms: wp.array[wp.transform], radii: wp.array[float]):
    """Conservative adjacency in each body's local frame, including convex seams."""
    return _shapes_adjacent(sa, ta, transforms, radii) and _shapes_adjacent(sb, tb, transforms, radii)


@wp.func
def _pose_motion(q: wp.array[wp.transform], previous_q: wp.array[wp.transform], body: int, point: wp.vec3):
    """Evaluate the rigid pose increment's SE(3) logarithm at a world point.

    The midpoint form avoids subtracting large world-origin moments. It is
    exact for a constant screw motion: a fixed pivot has zero displacement,
    and rolling does not mistake a material point's chord for tangential slip.
    """
    motion = wp.vec3(0.0)
    if body >= 0:
        current = q[body]
        previous = previous_q[body]
        x0 = wp.transform_get_translation(previous)
        x1 = wp.transform_get_translation(current)
        rotation = wp.transform_get_rotation(current) * wp.quat_inverse(wp.transform_get_rotation(previous))
        if rotation[3] < 0.0:
            rotation = -rotation
        vector = wp.vec3(rotation[0], rotation[1], rotation[2])
        sine = wp.length(vector)
        angular = 2.0 * vector
        translation = x1 - x0
        linear = translation
        if sine > 1.0e-6:
            half_angle = wp.atan2(sine, rotation[3])
            axis = vector / sine
            angular = (2.0 * half_angle) * axis
            scale = half_angle * rotation[3] / sine
            linear = scale * translation + (1.0 - scale) * wp.dot(axis, translation) * axis
        motion = linear + wp.cross(angular, point - 0.5 * (x0 + x1))
    return motion


@wp.func
def _carried_displacement(q: wp.array[wp.transform], prev: _PatchFrame, p: int, normal: wp.vec3):
    """Rotate stored spring displacement into the current tangent plane."""
    a = prev.body_a[p]
    error = prev.displacement[p]
    old_normal = prev.normal[p]
    if a >= 0:
        error = wp.transform_vector(q[a], error)
        old_normal = wp.transform_vector(q[a], old_normal)
    # Parallel transport preserves spring length when the tangent plane turns;
    # projection alone would dissipate history on every small rocking motion.
    cosine = wp.dot(old_normal, normal)
    if cosine > 0.0:
        axis = wp.cross(old_normal, normal)
        cross_error = wp.cross(axis, error)
        error += cross_error + wp.cross(axis, cross_error) / (1.0 + cosine)
    return error


_FLOOD_CUDA = """
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    int end = count.data[0];
    if (end > capacity) end = capacity;
    const int n_pairs = min(pair_count.data[0], pair_list.shape[0]);
    for (int pair = tile; pair < n_pairs; pair += n_tiles) {
    const int start = pair_list.data[pair];
    const wp::int64 key = keys.data[start];
    int stop = start + 1;
    while (stop < end && keys.data[stop] == key) ++stop;
    int remaining = stop - start;
    int member_stop = start;
    for (int index = start; index < stop; ++index) {
        const int seed = indices.data[index];
        __syncwarp();
        if (owner.data[seed] >= 0) continue;
        __syncwarp();
        if (lane == 0) owner.data[seed] = seed;
        remaining -= 1;
        int tail = seed;
        int first = (eligible.data[seed] != 0) ? seed : -1;
        const wp::vec3 seed_center = center.data[seed];
        const wp::vec3 seed_normal = normal.data[seed];
        const wp::vec2 seed_mu = mu.data[seed];
        const float r = radius.data[seed];
        int cursor = seed;
        int expanded_a = -2, expanded_b = -2;
        while (cursor >= 0 && remaining > 0) {
            const int sa = shape_a.data[cursor];
            const int sb = shape_b.data[cursor];
            if (sa != expanded_a || sb != expanded_b) {
                // The candidates' tests are independent: only an accepted contact changes
                // its own owner, so lanes test 32 of them at once and lane order applies
                // the accepted ones exactly as the serial scan would.
                for (int base = index + 1; base < stop; base += 32) {
                    const int j = base + lane;
                    int c = -1;
                    bool accept = false;
                    if (j < stop) {
                        c = indices.data[j];
                        if (owner.data[c] < 0) {
                            const wp::vec3 d = wp::sub(center.data[c], seed_center);
                            const wp::vec2 cmu = mu.data[c];
                            const bool compatible = wp::dot(seed_normal, normal.data[c]) >= 0.995f
                                && seed_mu[0] == cmu[0] && seed_mu[1] == cmu[1]
                                && wp::abs(wp::dot(d, seed_normal)) <= 0.02f * r
                                && wp::length_sq(d) <= 4.0f * r * r;
                            if (compatible && shapes_adjacent(sa, shape_a.data[c]) && shapes_adjacent(sb, shape_b.data[c]))
                                accept = true;
                        }
                    }
                    unsigned ballot = __ballot_sync(0xffffffffu, accept);
                    while (ballot) {
                        const int l = __ffs(ballot) - 1;
                        ballot &= ballot - 1u;
                        const int cc = __shfl_sync(0xffffffffu, c, l);
                        if (lane == 0) {
                            owner.data[cc] = seed;
                            next_contact.data[tail] = cc;
                        }
                        tail = cc;
                        remaining -= 1;
                        if (eligible.data[cc] != 0) {
                            if (first < 0) {
                                first = cc;
                            } else {
                                const wp::vec3 position = center.data[cc];
                                const wp::vec3 p0 = center.data[first];
                                if (position[0] < p0[0]
                                    || (position[0] == p0[0]
                                        && (position[1] < p0[1] || (position[1] == p0[1] && position[2] < p0[2]))))
                                    first = cc;
                            }
                        }
                    }
                }
                expanded_a = sa;
                expanded_b = sb;
            }
            __syncwarp();
            cursor = next_contact.data[cursor];
        }
        // Members in sorted order with the member farthest from the first location,
        // as _build collects them: the same independent test, applied in lane order.
        __syncwarp();
        const bool carry_only = first < 0;
        const int anchor = carry_only ? seed : first;
        const wp::vec3 anchor_center = center.data[anchor];
        int second = anchor;
        float separation = 0.0f;
        for (int base = index; base < stop; base += 32) {
            const int j = base + lane;
            const int c = (j < stop) ? indices.data[j] : -1;
            unsigned ballot = __ballot_sync(0xffffffffu, c >= 0 && owner.data[c] == seed);
            while (ballot) {
                const int l = __ffs(ballot) - 1;
                ballot &= ballot - 1u;
                const int cc = __shfl_sync(0xffffffffu, c, l);
                if (lane == 0) members.data[member_stop] = cc;
                member_stop += 1;
                if (carry_only || eligible.data[cc] != 0) {
                    const float distance = wp::length_sq(wp::sub(center.data[cc], anchor_center));
                    if (distance > separation) {
                        separation = distance;
                        second = cc;
                    }
                }
            }
        }
        if (lane == 0) {
            next_contact.data[tail] = seed;
            first_location.data[seed] = first;
            region_stop.data[seed] = member_stop;
            region_second.data[seed] = second;
            region_separation.data[seed] = separation;
        }
    }
    }
#endif
"""
_FLOOD_CUDA = _FLOOD_CUDA.replace(
    "#if defined(__CUDA_ARCH__)",
    """#if defined(__CUDA_ARCH__)
    const auto shapes_adjacent = [&](int a, int b) {
        if (a == b) return true;
        if (a < 0 || b < 0) return false;
        const float rad = shape_radius.data[a] + shape_radius.data[b];
        const wp::vec3 delta = wp::sub(
            wp::transform_get_translation(shape_transform.data[a]), wp::transform_get_translation(shape_transform.data[b]));
        return wp::length_sq(delta) <= (1.0f + 1.0e-5f) * rad * rad;
    };""",
    1,
)


@wp.func_native(_FLOOD_CUDA)
def _flood_native(
    tile: int,
    n_tiles: int,
    capacity: int,
    pair_list: wp.array[int],
    pair_count: wp.array[int],
    count: wp.array[int],
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    owner: wp.array[int],
    eligible: wp.array[int],
    center: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    mu: wp.array[wp.vec2],
    radius: wp.array[float],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_radius: wp.array[float],
    next_contact: wp.array[int],
    first_location: wp.array[int],
    members: wp.array[int],
    region_stop: wp.array[int],
    region_second: wp.array[int],
    region_separation: wp.array[float],
): ...


_FLOOD_MIN_CONTACTS = 32
"""Body pairs with more contacts than this grow their regions on a warp; smaller pairs in ``_build``."""


@wp.kernel(enable_backward=False)
def _list_flood_pairs(
    capacity: int,
    count: wp.array[int],
    keys: wp.array[wp.int64],
    min_contacts: int,
    frozen_bodies: wp.array[int],
    frame: _PatchFrame,
    pair_list: wp.array[int],
    pair_count: wp.array[int],
):
    """List the sorted body pairs with more than ``min_contacts`` contacts for :func:`_flood_regions`."""
    start = wp.tid()
    end = wp.min(count[0], capacity)
    if start >= end or (start > 0 and keys[start - 1] == keys[start]):
        return
    key = keys[start]
    if key == wp.int64(0x7FFFFFFFFFFFFFFF):
        return
    if start + min_contacts >= end or keys[start + min_contacts] != key:
        return
    # Frozen pairs carry their history in ``_build`` and flood there only if the carry fails.
    if _pair_frozen(frozen_bodies, frame, start):
        return
    index = wp.atomic_add(pair_count, 0, 1)
    if index < pair_list.shape[0]:
        pair_list[index] = start


@wp.kernel(enable_backward=False)
def _flood_regions(
    n_tiles: int,
    capacity: int,
    pair_list: wp.array[int],
    pair_count: wp.array[int],
    count: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_radius: wp.array[float],
    frame: _PatchFrame,
    patches: FrictionPatches,
):
    """Grow the listed body pairs' friction regions exactly as ``_build`` would, one warp each.

    ``_build`` grows a pair's regions on one thread, testing every later contact of the
    pair for each new convex piece of a region, which is quadratic in the pair's contact
    count and runs serially for the largest pair. The tests for one scan are independent,
    so the warp evaluates them 32 at a time and applies the accepted contacts in scan
    order: owners, region links and each region's first friction location match the
    serial flood exactly. The same holds for each region's member list, collected right
    after its flood. A fixed grid of warps strides over the pairs listed by
    :func:`_list_flood_pairs`, so the launch does not scale with contact capacity.
    CUDA only; ``_build`` floods the other pairs, and all pairs on the CPU.
    """
    tile, _lane = wp.tid()
    _flood_native(
        tile,
        n_tiles,
        capacity,
        pair_list,
        pair_count,
        count,
        frame.keys,
        frame.indices,
        frame.owner,
        frame.eligible,
        frame.center,
        frame.normal,
        frame.mu,
        frame.radius,
        frame.shape_a,
        frame.shape_b,
        shape_transform,
        shape_radius,
        patches.next_contact,
        frame.first,
        frame.members,
        frame.region_stop,
        frame.second,
        frame.separation,
    )


# Capped at the register count of the single-path kernel: the device-flood branch
# otherwise raises it from 96 to 120, and the lost occupancy costs more than the
# branch saves when a batch of small worlds floods every pair here. The cap needs
# Warp's cuda_max_registers (1.17); older Warp builds the uncapped kernel.
_BUILD_KERNEL_OPTIONS = (
    {"cuda_max_registers": 96} if "cuda_max_registers" in inspect.signature(wp.kernel).parameters else {}
)


@wp.kernel(enable_backward=False, **_BUILD_KERNEL_OPTIONS)
def _build(
    count: wp.array[int],
    q: wp.array[wp.transform],
    previous_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_radius: wp.array[float],
    frozen_bodies: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    patches: FrictionPatches,
    flooded: int,
):
    """One thread per sorted body pair; writes disjoint current/previous records.

    With ``flooded``, pairs with more than ``_FLOOD_MIN_CONTACTS`` contacts were grown by
    :func:`_flood_regions`; their seeds own themselves and each region's first location,
    member range and farthest member are stored. Other pairs flood here.
    """
    start = wp.tid()
    end = wp.min(count[0], frame.center.shape[0])
    if start >= end or (start > 0 and frame.keys[start - 1] == frame.keys[start]):
        return
    key = frame.keys[start]
    if key == wp.int64(0x7FFFFFFFFFFFFFFF):
        return
    stop = start + 1
    while stop < end and frame.keys[stop] == key:
        stop += 1

    # Binary search the previous pair, avoiding a scan over other worlds/pairs.
    lo = int(0)
    hi = prev.center.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if prev.keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    prev_start = lo
    prev_stop = lo
    # Index valid history once per pair rather than rescanning all old contacts
    # for every new friction location. Keep sorted order for matching ties.
    prev_anchor_stop = prev_start
    while prev_stop < prev.center.shape[0] and prev.keys[prev_stop] == key:
        p = prev.indices[prev_stop]
        prev.used[p] = 0
        if prev.valid[p] != 0:
            prev.members[prev_anchor_stop] = p
            prev_anchor_stop += 1
        prev_stop += 1
    if _carry_frozen_pair(q, frozen_bodies, frame, prev, start, stop, prev_start, prev_stop):
        return

    member_stop = start
    remaining = stop - start
    # Pairs with more than _FLOOD_MIN_CONTACTS contacts were grown by _flood_regions.
    pair_flooded = int(0)
    if flooded != 0 and remaining > _FLOOD_MIN_CONTACTS and not _pair_frozen(frozen_bodies, frame, start):
        pair_flooded = 1
    for index in range(start, stop):
        seed = frame.indices[index]
        first = int(-1)
        if pair_flooded != 0:
            if frame.owner[seed] != seed:
                continue
            first = frame.first[seed]
        else:
            if frame.owner[seed] >= 0:
                continue
            frame.owner[seed] = seed
            remaining -= 1
            tail = seed
            if frame.eligible[seed] != 0:
                first = seed
            cursor = seed
            expanded_a = int(-2)
            expanded_b = int(-2)
            # Flood through adjacent convex pieces. The queue is the contact list
            # itself; no allocation, atomics, or fixed patch-count limit is needed.
            while cursor >= 0 and remaining > 0:
                sa = frame.shape_a[cursor]
                sb = frame.shape_b[cursor]
                if sa != expanded_a or sb != expanded_b:
                    for j in range(index + 1, stop):
                        c = frame.indices[j]
                        if frame.owner[c] < 0 and _compatible(frame, seed, c):
                            if _geometry_adjacent(
                                sa, sb, frame.shape_a[c], frame.shape_b[c], shape_transform, shape_radius
                            ):
                                frame.owner[c] = seed
                                remaining -= 1
                                patches.next_contact[tail] = c
                                tail = c
                                if frame.eligible[c] != 0:
                                    if first < 0:
                                        first = c
                                    else:
                                        position = frame.center[c]
                                        p0 = frame.center[first]
                                        if position[0] < p0[0] or (
                                            position[0] == p0[0]
                                            and (position[1] < p0[1] or (position[1] == p0[1] and position[2] < p0[2]))
                                        ):
                                            first = c
                    expanded_a = sa
                    expanded_b = sb
                cursor = patches.next_contact[cursor]
            patches.next_contact[tail] = seed
        carry_only = int(0)
        if first < 0:
            # No member may carry friction rows this step (gap filters). Keep the
            # region's anchor history on its members with zero weight so a single
            # filtered step does not forget the accumulated positional correction.
            carry_only = int(1)
            first = seed
        second = first
        separation = float(0.0)
        member_start = member_stop
        if pair_flooded != 0:
            member_stop = frame.region_stop[seed]
            second = frame.second[seed]
            separation = frame.separation[seed]
        else:
            # Preserve sorted contact order for footprint sums and tie breaking.
            # The collision list follows flood order and must remain unchanged.
            for j in range(index, stop):
                c = frame.indices[j]
                if frame.owner[c] != seed:
                    continue
                frame.members[member_stop] = c
                member_stop += 1
                if carry_only != 0 or frame.eligible[c] != 0:
                    distance = wp.length_sq(frame.center[c] - frame.center[first])
                    if distance > separation:
                        separation = distance
                        second = c
        anchors = int(1)
        if separation > 1.0e-8 * frame.radius[seed] * frame.radius[seed]:
            anchors = 2
        location0 = frame.center[first]
        location1 = frame.center[second]
        if anchors == 2 and carry_only == 0:
            # Use the footprint's principal extent on its principal line through
            # the centroid. Picking opposite corners of a narrow rectangle
            # introduces an artificial diagonal friction couple even when the
            # footprint is symmetric.
            origin = location0
            t0, t1 = contact_tangent_basis(frame.normal[seed])
            # Accumulate moments in double precision: contact-order roundoff
            # must not give an isotropic footprint a spurious principal axis.
            # Raw moments also avoid a separate pass to compute the centroid.
            sum_x = wp.float64(0.0)
            sum_y = wp.float64(0.0)
            sum_xx = wp.float64(0.0)
            sum_xy = wp.float64(0.0)
            sum_yy = wp.float64(0.0)
            sum_center = wp.vec3d(0.0)
            members = int(0)
            for j in range(member_start, member_stop):
                c = frame.members[j]
                if frame.eligible[c] != 0:
                    offset = frame.center[c] - origin
                    x = wp.float64(wp.dot(offset, t0))
                    y = wp.float64(wp.dot(offset, t1))
                    sum_x += x
                    sum_y += y
                    sum_xx += x * x
                    sum_xy += x * y
                    sum_yy += y * y
                    point = frame.center[c]
                    sum_center += wp.vec3d(wp.float64(point[0]), wp.float64(point[1]), wp.float64(point[2]))
                    members += 1
            member_count = wp.float64(members)
            xx = float(sum_xx - sum_x * sum_x / member_count)
            xy = float(sum_xy - sum_x * sum_y / member_count)
            yy = float(sum_yy - sum_y * sum_y / member_count)
            spread = wp.sqrt((xx - yy) * (xx - yy) + 4.0 * xy * xy)
            # An isotropic footprint has no distinguished axis; retain its
            # canonical farthest pair instead of amplifying roundoff.
            if spread > 1.0e-6 * (xx + yy):
                largest = 0.5 * (xx + yy + spread)
                axis2 = wp.vec2(xy, largest - xx)
                if xx >= yy:
                    axis2 = wp.vec2(largest - yy, xy)
                axis2 = wp.normalize(axis2)
                axis = axis2[0] * t0 + axis2[1] * t1
                low = float(1.0e30)
                high = float(-1.0e30)
                for j in range(member_start, member_stop):
                    c = frame.members[j]
                    if frame.eligible[c] != 0:
                        projection = wp.dot(frame.center[c] - origin, axis)
                        if projection < low:
                            low = projection
                            first = c
                        if projection > high:
                            high = projection
                            second = c
                # Project the axial extremes onto the principal line through the
                # centroid instead of averaging the members near each end. Any
                # grouping tolerance fails for some misalignment: a slight shear
                # spreads an edge's projections, keeps one corner per edge and
                # restores the diagonal couple. The projection selects no members,
                # so anchors move continuously with the contacts and stay on a
                # symmetric footprint's axis of symmetry.
                mean = sum_center / member_count
                centroid = wp.vec3(float(mean[0]), float(mean[1]), float(mean[2]))
                location0 = centroid + wp.dot(frame.center[first] - centroid, axis) * axis
                location1 = centroid + wp.dot(frame.center[second] - centroid, axis) * axis
        for aidx in range(anchors):
            c = first
            if aidx == 1:
                c = second
            location = location0
            if aidx == 1:
                location = location1
            a = frame.body_a[c]
            b = frame.body_b[c]
            n = frame.normal[c]
            r = frame.radius[c]
            surface_a = frame.surface_a[c]
            surface_b = frame.surface_b[c]
            current_gap = wp.dot(_world_point(q, a, surface_a) - _world_point(q, b, surface_b), n)
            chosen = int(-1)
            nearest = float(1.0e30)
            motion = wp.vec3(0.0)
            if prev_start < prev_stop:
                motion = _pose_motion(q, previous_q, a, location) - _pose_motion(q, previous_q, b, location)
            for j in range(prev_start, prev_anchor_stop):
                p = prev.members[j]
                if (
                    prev.valid[p] == 0
                    or prev.used[p] != 0
                    or prev.mu[p][0] != frame.mu[c][0]
                    or prev.mu[p][1] != frame.mu[c][1]
                ):
                    continue
                old_n = prev.normal[p]
                if a >= 0:
                    old_n = wp.transform_vector(q[a], old_n)
                # Reject incompatible normals and positions before transporting
                # the spring or evaluating support witnesses. These are the same
                # gates as below, ordered to avoid work for rejected history.
                if not (wp.dot(old_n, n) >= 0.995):
                    continue
                pa = _world_point(q, a, prev.anchor_a[p])
                pb = _world_point(q, b, prev.anchor_b[p])
                delta = pa - pb
                offset = 0.5 * (pa + pb) - location
                distance = wp.length_sq(offset)
                if not (
                    distance < nearest
                    and distance <= 4.0 * r * r
                    and wp.abs(wp.dot(delta, n)) <= 0.1 * r
                    and wp.abs(wp.dot(offset, n)) <= 0.05 * r
                ):
                    continue
                # Honor the existing contact envelope: a zero-gap test drops
                # valid speculative contacts during a squeeze. Surface
                # witnesses retain penetration during decompression and
                # reject rocking beyond the same envelope as fresh contacts.
                support_gap = wp.dot(_world_point(q, a, prev.surface_a[p]) - _world_point(q, b, prev.surface_b[p]), n)
                if not (support_gap <= wp.max(current_gap, frame.support_gap_limit[c]) + 1.0e-5 * r):
                    continue
                error = _carried_displacement(q, prev, p, n)
                error += motion
                tangent_delta = error - n * wp.dot(error, n)
                # The tangential gate bounds uncorrected slip of a carried pair at a
                # tenth of the smaller body's radius, about three times the measured
                # Baumgarte equilibrium separation of a held box at 60 Hz. Tighter
                # gates re-anchor below that equilibrium and turn the correction
                # into creep; genuine sliding is released by ``finish_patch_impulses``.
                if not (wp.length_sq(tangent_delta) <= 0.01 * r * r):
                    continue
                connected = _geometry_adjacent(
                    prev.shape_a[p], prev.shape_b[p], frame.shape_a[c], frame.shape_b[c], shape_transform, shape_radius
                )
                if not connected:
                    for k in range(member_start, member_stop):
                        member = frame.members[k]
                        if _geometry_adjacent(
                            prev.shape_a[p],
                            prev.shape_b[p],
                            frame.shape_a[member],
                            frame.shape_b[member],
                            shape_transform,
                            shape_radius,
                        ):
                            connected = True
                            break
                if connected:
                    chosen = p
                    nearest = distance
            pa = location
            pb = pa
            anchor_a = _local_point(q, a, pa)
            anchor_b = _local_point(q, b, pb)
            carried_impulse = wp.vec3(0.0)
            error = wp.vec3(0.0)
            if chosen >= 0:
                prev.used[chosen] = 1
                error = _carried_displacement(q, prev, chosen, n)
                old_center = 0.5 * (
                    _world_point(q, a, prev.anchor_a[chosen]) + _world_point(q, b, prev.anchor_b[chosen])
                )
                # The two spring samples describe a planar rigid displacement:
                # translation plus twist. Evaluate that field at the new points
                # instead of transferring old lever-arm errors independently.
                # A one-point region carries translation only.
                for j in range(prev_start, prev_anchor_stop):
                    partner = prev.members[j]
                    if partner != chosen and prev.valid[partner] != 0 and prev.owner[partner] == prev.owner[chosen]:
                        other_normal = prev.normal[partner]
                        if a >= 0:
                            other_normal = wp.transform_vector(q[a], other_normal)
                        other_gap = wp.dot(
                            _world_point(q, a, prev.surface_a[partner]) - _world_point(q, b, prev.surface_b[partner]), n
                        )
                        if (
                            wp.dot(other_normal, n) < 0.995
                            or other_gap > wp.max(current_gap, frame.support_gap_limit[c]) + 1.0e-5 * r
                        ):
                            break
                        other_center = 0.5 * (
                            _world_point(q, a, prev.anchor_a[partner]) + _world_point(q, b, prev.anchor_b[partner])
                        )
                        span = other_center - old_center
                        span -= n * wp.dot(span, n)
                        span_sq = wp.length_sq(span)
                        if span_sq > 1.0e-8 * r * r:
                            other_error = _carried_displacement(q, prev, partner, n)
                            spin = wp.dot(other_error - error, wp.cross(n, span)) / span_sq
                            error = 0.5 * (error + other_error) + spin * wp.cross(
                                n, pa - 0.5 * (old_center + other_center)
                            )
                        break
                error += motion
                error -= n * wp.dot(error, n)
                carried_impulse = prev.tangent_impulse[chosen]
            stored_error = error
            if a >= 0:
                stored_error = wp.transform_vector(wp.transform_inverse(q[a]), error)
            frame.displacement[c] = stored_error
            frame.source[c] = chosen
            if carry_only != 0 and chosen < 0:
                # Filtered members only carry existing history; they never start
                # it, because no friction row held the pair during this step.
                continue
            frame.anchor_a[c] = anchor_a
            frame.anchor_b[c] = anchor_b
            frame.surface_a[c] = surface_a
            frame.surface_b[c] = surface_b
            # Default for anchors without friction rows this step; solved rows
            # overwrite it in ``finish_patch_impulses``.
            frame.tangent_impulse[c] = carried_impulse
            frame.valid[c] = 1
            patches.weight[c] = 1.0 / float(anchors)
            if carry_only != 0:
                patches.weight[c] = 0.0
            if frame.flipped[c] != 0:
                temp = pa
                pa = pb
                pb = temp
                n = -n
            patches.point_a[c] = pa
            patches.point_b[c] = pb
            t0, t1 = contact_tangent_basis(n)
            if frame.flipped[c] != 0:
                error = -error
            patches.phi[c] = wp.vec2(wp.dot(t0, error), wp.dot(t1, error))


@wp.func
def _pair_frozen(frozen_bodies: wp.array[int], frame: _PatchFrame, start: int):
    """Whether either body of the sorted pair starting at ``start`` is frozen."""
    if frozen_bodies.shape[0] == 0:
        return False
    a = frame.body_a[frame.indices[start]]
    b = frame.body_b[frame.indices[start]]
    return (a >= 0 and frozen_bodies[a] != 0) or (b >= 0 and frozen_bodies[b] != 0)


@wp.func
def _carry_frozen_pair(
    q: wp.array[wp.transform],
    frozen_bodies: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    start: int,
    stop: int,
    prev_start: int,
    prev_stop: int,
):
    """Carry a frozen pair's history unchanged; its poses and contacts repeat and it has no rows."""
    if prev_stop - prev_start != stop - start or not _pair_frozen(frozen_bodies, frame, start):
        return False
    a = frame.body_a[frame.indices[start]]
    for k in range(stop - start):
        c = frame.indices[start + k]
        p = prev.indices[prev_start + k]
        if frame.shape_a[c] != prev.shape_a[p] or frame.shape_b[c] != prev.shape_b[p]:
            return False
    for k in range(stop - start):
        c = frame.indices[start + k]
        p = prev.indices[prev_start + k]
        frame.source[c] = p
        frame.valid[c] = prev.valid[p]
        frame.displacement[c] = prev.displacement[p]
        frame.tangent_impulse[c] = prev.tangent_impulse[p]
        frame.anchor_a[c] = prev.anchor_a[p]
        frame.anchor_b[c] = prev.anchor_b[p]
        if prev.valid[p] != 0:
            frame.surface_a[c] = prev.surface_a[p]
            frame.surface_b[c] = prev.surface_b[p]
            normal = prev.normal[p]
            if a >= 0:
                normal = wp.transform_vector(q[a], normal)
            frame.normal[c] = normal
        for j in range(stop - start):
            if prev.indices[prev_start + j] == prev.owner[p]:
                frame.owner[c] = frame.indices[start + j]
    return True


@wp.kernel(enable_backward=False)
def _store_history(
    q: wp.array[wp.transform],
    body_world: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    previous_world: wp.array[int],
    previous_q: wp.array[wp.transform],
):
    """Carry contact history and input poses, independent of solver velocity passes."""
    c = wp.tid()
    if c < previous_q.shape[0]:
        previous_q[c] = q[c]
    if c >= frame.valid.shape[0]:
        return
    normal = frame.normal[c]
    if frame.valid[c] != 0 and frame.body_a[c] >= 0:
        normal = wp.transform_vector(wp.transform_inverse(q[frame.body_a[c]]), normal)
    prev.keys[c] = frame.keys[c]
    prev.indices[c] = frame.indices[c]
    prev.normal[c] = normal
    prev.displacement[c] = frame.displacement[c]
    prev.mu[c] = frame.mu[c]
    prev.body_a[c] = frame.body_a[c]
    prev.body_b[c] = frame.body_b[c]
    prev.owner[c] = frame.owner[c]
    prev.shape_a[c] = frame.shape_a[c]
    prev.shape_b[c] = frame.shape_b[c]
    prev.anchor_a[c] = frame.anchor_a[c]
    prev.anchor_b[c] = frame.anchor_b[c]
    prev.surface_a[c] = frame.surface_a[c]
    prev.surface_b[c] = frame.surface_b[c]
    prev.valid[c] = frame.valid[c]
    prev.tangent_impulse[c] = frame.tangent_impulse[c]
    world = int(-1)
    if frame.valid[c] != 0:
        if frame.body_a[c] >= 0:
            world = body_world[frame.body_a[c]]
        if world < 0 and frame.body_b[c] >= 0:
            world = body_world[frame.body_b[c]]
        # The solver maps the default, unpartitioned world (-1) to world zero.
        world = wp.max(world, 0)
    previous_world[c] = world


@wp.kernel(enable_backward=False)
def _invalidate_geometry_history(prev: _PatchFrame, bodies: wp.array[int], shapes: wp.array[int]):
    """Discard material points whose supporting geometry changed."""
    c = wp.tid()
    if prev.valid[c] == 0:
        return
    changed = bool(False)
    if prev.shape_a[c] >= 0:
        changed = shapes[prev.shape_a[c]] != 0
    if prev.shape_b[c] >= 0:
        changed = changed or shapes[prev.shape_b[c]] != 0
    if prev.body_a[c] >= 0:
        changed = changed or bodies[prev.body_a[c]] != 0
    if prev.body_b[c] >= 0:
        changed = changed or bodies[prev.body_b[c]] != 0
    if changed:
        prev.valid[c] = 0


class _FrictionPatchState:
    """Own preallocated patch frames; never read device counters on the host."""

    def __init__(self, model, capacity, enabled, phi):
        self.view = FrictionPatches()
        self.view.enabled = int(enabled)
        self.view.phi = phi
        self.capacity = capacity
        device = model.device
        n = capacity if enabled else 0
        self.view.weight = wp.zeros(n, dtype=float, device=device)
        self.view.next_contact = wp.full(n, -1, dtype=int, device=device)
        self.view.point_a = wp.zeros(n, dtype=wp.vec3, device=device)
        self.view.point_b = wp.zeros(n, dtype=wp.vec3, device=device)
        if not enabled:
            return
        self.body_world = model.body_world
        self.body_radius = wp.zeros(model.body_count, dtype=float, device=device)
        self.update_geometry(model)
        self.current = self._frame(capacity, device)
        self.previous = self._frame(capacity, device)
        self.previous_world = wp.full(capacity, -1, dtype=int, device=device)
        self.previous_q = wp.zeros(model.body_count, dtype=wp.transform, device=device)
        # Body pairs large enough for the warp flood: at most one per _FLOOD_MIN_CONTACTS + 1
        # contacts, grown by a fixed grid of warps so the launch does not scale with capacity.
        pair_capacity = capacity // (_FLOOD_MIN_CONTACTS + 1) + 1
        self._flood_pair_list = wp.zeros(pair_capacity, dtype=int, device=device)
        self._flood_pair_count = wp.zeros(1, dtype=int, device=device)
        self._flood_tiles = min(pair_capacity, 1024)
        self._no_frozen_bodies = wp.zeros(0, dtype=int, device=device)

    def update_geometry(self, model):
        """Refresh scales and retire affected history after explicit geometry edits."""
        geometry = {
            name: getattr(model, name).numpy().copy()
            for name in (
                "shape_body",
                "shape_transform",
                "shape_scale",
                "shape_type",
                "shape_source_ptr",
                "shape_margin",
                "shape_is_solid",
            )
        }
        radii = np.zeros(model.body_count, dtype=np.float32)
        for body, radius, transform in zip(
            geometry["shape_body"], model.shape_collision_radius.numpy(), geometry["shape_transform"], strict=True
        ):
            if body >= 0:
                radii[body] = max(radii[body], radius + np.linalg.norm(transform[:3]))
        self.body_radius.assign(radii)
        if hasattr(self, "_geometry"):
            changed = np.zeros(len(geometry["shape_body"]), dtype=bool)
            for name, values in geometry.items():
                difference = values != self._geometry[name]
                if difference.ndim > 1:
                    difference = np.any(difference, axis=tuple(range(1, difference.ndim)))
                changed |= difference
            if np.any(changed):
                # An anchor may have crossed a convex seam since it was created.
                # Invalidate the affected body's history, including other carrier
                # shapes, while keeping unrelated bodies and static shapes intact.
                bodies = np.zeros(model.body_count, dtype=np.int32)
                for shape_body in (self._geometry["shape_body"], geometry["shape_body"]):
                    affected = shape_body[changed]
                    bodies[affected[affected >= 0]] = 1
                wp.launch(
                    _invalidate_geometry_history,
                    dim=self.capacity,
                    inputs=[
                        self.previous,
                        wp.array(bodies, dtype=int, device=model.device),
                        wp.array(changed, dtype=int, device=model.device),
                    ],
                    device=model.device,
                )
        self._geometry = geometry

    @staticmethod
    def _frame(n, device):
        frame = _PatchFrame()
        frame.keys = wp.full(2 * n, 0x7FFFFFFFFFFFFFFF, dtype=wp.int64, device=device)
        frame.indices = wp.zeros(2 * n, dtype=int, device=device)
        for field in (
            "center",
            "normal",
            "displacement",
            "anchor_a",
            "anchor_b",
            "surface_a",
            "surface_b",
            "tangent_impulse",
        ):
            setattr(frame, field, wp.zeros(n, dtype=wp.vec3, device=device))
        frame.mu = wp.zeros(n, dtype=wp.vec2, device=device)
        frame.radius = wp.zeros(n, dtype=float, device=device)
        frame.support_gap_limit = wp.zeros(n, dtype=float, device=device)
        frame.separation = wp.zeros(n, dtype=float, device=device)
        for field in (
            "body_a",
            "body_b",
            "shape_a",
            "shape_b",
            "flipped",
            "owner",
            "valid",
            "used",
            "source",
            "eligible",
            "members",
            "first",
            "region_stop",
            "second",
        ):
            setattr(frame, field, wp.zeros(n, dtype=int, device=device))
        return frame

    def build(
        self,
        model,
        state,
        contacts,
        *,
        body_to_articulation=None,
        is_free_rigid=None,
        contact_gap_gate=0.0,
        same_articulation_gap_gate=0.0,
        articulation_pair_gap_gate=0.0,
        friction_gap=float("inf"),
        friction_articulation_pairs_only=False,
        frozen_bodies=None,
    ):
        wp.launch(
            _prepare,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                model.shape_body,
                model.shape_material_mu,
                model.shape_gap,
                self.body_radius,
                state.body_q,
                body_to_articulation,
                is_free_rigid,
                contact_gap_gate,
                same_articulation_gap_gate,
                articulation_pair_gap_gate,
                friction_gap,
                int(friction_articulation_pairs_only),
                self.current,
                self.view,
            ],
            device=model.device,
        )
        wp.utils.radix_sort_pairs(self.current.keys, self.current.indices, self.capacity)
        # The warp-per-pair flood matches the serial one exactly; the switch exists so
        # tests can compare them.
        flooded = model.device.is_cuda and getattr(self, "device_flood", True)
        if flooded:
            self._flood_pair_count.zero_()
            wp.launch(
                _list_flood_pairs,
                dim=self.capacity,
                inputs=[
                    self.capacity,
                    contacts.rigid_contact_count,
                    self.current.keys,
                    _FLOOD_MIN_CONTACTS,
                    frozen_bodies if frozen_bodies is not None else self._no_frozen_bodies,
                    self.current,
                    self._flood_pair_list,
                    self._flood_pair_count,
                ],
                device=model.device,
            )
            wp.launch_tiled(
                _flood_regions,
                dim=[self._flood_tiles],
                inputs=[
                    self._flood_tiles,
                    self.capacity,
                    self._flood_pair_list,
                    self._flood_pair_count,
                    contacts.rigid_contact_count,
                    model.shape_transform,
                    model.shape_collision_radius,
                    self.current,
                    self.view,
                ],
                block_dim=32,
                device=model.device,
            )
        wp.launch(
            _build,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                state.body_q,
                self.previous_q,
                model.shape_transform,
                model.shape_collision_radius,
                frozen_bodies if frozen_bodies is not None else self._no_frozen_bodies,
                self.current,
                self.previous,
                self.view,
                int(flooded),
            ],
            device=model.device,
        )

    def store(self, state):
        wp.launch(
            _store_history,
            dim=max(self.capacity, self.previous_q.shape[0]),
            inputs=[
                state.body_q,
                self.body_world,
                self.current,
                self.previous,
                self.previous_world,
                self.previous_q,
            ],
            device=state.body_q.device,
        )


@wp.func
def patch_normal_load(parents: wp.array2d[int], impulses: wp.array2d[float], world: int, first: int):
    """Sum a region's live normal impulses; -1 terminates ordinary point contacts.

    A contact row's parent field links the next normal in its region (a circular
    list). Friction rows still point to their adjacent normal row. Other row
    families retain their original parent semantics.
    """
    load = impulses[world, first]
    row = parents[world, first]
    while row >= 0 and row != first:
        load += impulses[world, row]
        row = parents[world, row]
    return load


@wp.kernel(enable_backward=False)
def link_patch_rows(
    count: wp.array[int],
    patches: FrictionPatches,
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
):
    """Link only allocated rows and divide the Coulomb budget among surviving anchors.

    Row builders write the unscaled friction coefficient; this kernel divides it by
    the number of anchors that actually received rows, so the pooled normal load
    is shared exactly once.
    """
    c = wp.tid()
    if c >= count[0] or slot[c] < 0 or path[c] != route:
        return
    row = patches.next_contact[c]
    if row < 0:
        return
    next_slot = slot[c]
    anchors = int(patches.weight[c] > 0.0 and slots_needed[c] == 3)
    while row != c:
        if slot[row] >= 0 and path[row] == route and world[row] == world[c]:
            if next_slot == slot[c]:
                next_slot = slot[row]
            if patches.weight[row] > 0.0 and slots_needed[row] == 3:
                anchors += 1
            if patches.weight[c] == 0.0:
                break
        row = patches.next_contact[row]
    parents[world[c], slot[c]] = next_slot
    if patches.weight[c] > 0.0 and slots_needed[c] == 3 and anchors > 0:
        for t in range(1, 3):
            mu[world[c], slot[c] + t] /= float(anchors)


@wp.func
def contact_tangent_basis(n: wp.vec3):
    """Return the deterministic tangent pair every FeatherPGS friction row uses for ``n``."""
    tangent0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
    if wp.length_sq(tangent0) < 1.0e-12:
        tangent0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    tangent0 = wp.normalize(tangent0)
    tangent1 = wp.normalize(wp.cross(n, tangent0))
    return tangent0, tangent1


@wp.kernel(enable_backward=False)
def seed_patch_impulses(
    count: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    q: wp.array[wp.transform],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    scale: float,
):
    """Transport cached patch impulses after all contact normals have been seeded.

    Anchors without patch history keep whatever the contact-matched warm start
    seeded; only carried anchors overwrite their friction rows.
    """
    c = wp.tid()
    if c >= count[0] or path[c] != route or slot[c] < 0 or slots_needed[c] != 3:
        return
    source = frame.source[c]
    if source < 0:
        return
    tangent = prev.tangent_impulse[source] * scale
    if frame.body_a[c] >= 0:
        tangent = wp.transform_vector(q[frame.body_a[c]], tangent)
    n = frame.normal[c]
    if frame.flipped[c] != 0:
        n = -n
        tangent = -tangent
    t0, t1 = contact_tangent_basis(n)
    value = wp.vec2(wp.dot(tangent, t0), wp.dot(tangent, t1))
    w = world[c]
    s = slot[c]
    radius = wp.max(mu[w, s + 1] * patch_normal_load(parents, impulses, w, s), 0.0)
    magnitude = wp.length(value)
    if magnitude > radius and magnitude > 0.0:
        value *= radius / magnitude
    impulses[w, s + 1] = value[0]
    impulses[w, s + 2] = value[1]


@wp.func
def _point_velocity(
    q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3], body: int, anchor: wp.vec3
):
    velocity = wp.vec3(0.0)
    if body >= 0:
        velocity = velocity_at_point(qd[body], wp.transform_vector(q[body], anchor - com[body]))
    return velocity


@wp.kernel(enable_backward=False)
def finish_patch_impulses(
    count: wp.array[int],
    frame: _PatchFrame,
    q: wp.array[wp.transform],
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    dt: float,
):
    """Cache solved friction and release saturated anchors with tangential motion.

    The tolerance is relative to geometry and timestep. Merely resting at the
    Coulomb limit does not prove sliding, so it must not erase static history.
    """
    c = wp.tid()
    if c >= count[0] or frame.valid[c] == 0:
        return
    if slot[c] < 0:
        # Capacity rejection provides no solved normal load; preserve the
        # geometrically supported history until rows are available again.
        return
    if path[c] != route:
        return
    w = world[c]
    s = slot[c]
    load = patch_normal_load(parents, impulses, w, s)
    if load <= 0.0:
        # A speculative region cannot accumulate a spring while unsupported
        # and apply that error when it later lands (common on curved meshes).
        frame.valid[c] = 0
        return
    if slots_needed[c] != 3:
        # Friction was filtered, but its normal rows still prove support.
        return
    n = frame.normal[c]
    t0, t1 = contact_tangent_basis(n)
    if frame.flipped[c] != 0:
        t0, t1 = contact_tangent_basis(-n)
    tangent = impulses[w, s + 1] * t0 + impulses[w, s + 2] * t1
    if frame.flipped[c] != 0:
        tangent = -tangent
    stored = tangent
    a = frame.body_a[c]
    b = frame.body_b[c]
    if a >= 0:
        stored = wp.transform_vector(wp.transform_inverse(q[a]), tangent)
    frame.tangent_impulse[c] = stored
    radius = wp.max(mu[w, s + 1] * load, 0.0)
    speed = _point_velocity(q_out, qd_out, com, a, frame.anchor_a[c]) - _point_velocity(
        q_out, qd_out, com, b, frame.anchor_b[c]
    )
    # Check opposing tangential motion to distinguish slipping from impending slip.
    if radius > 0.0 and wp.length(tangent) >= (1.0 - 1.0e-5) * radius:
        if wp.dot(speed, tangent) < -radius * (1.0e-5 * frame.radius[c] / dt):
            frame.valid[c] = 0
