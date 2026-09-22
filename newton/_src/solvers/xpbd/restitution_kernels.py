# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...math import velocity_at_point
from ...sim.contacts import contact_surface_point, contact_surface_separation


@wp.kernel
def mark_restitution_contacts(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    # output
    restitution_contact_active: wp.array[wp.int32],
):
    """Mark contacts that penetrate during position projection."""
    tid = wp.tid()

    if tid >= contact_count[0]:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    if body_a == body_b:
        return

    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    bx_a = wp.transform_point(X_wb_a, contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, contact_point1[tid])
    d = contact_surface_separation(bx_a, bx_b, contact_normal[tid], contact_thickness0[tid], contact_thickness1[tid])
    if d < 0.0:
        restitution_contact_active[tid] = 1


# Maximum number of contacts the restitution solve keeps per manifold.
# Manifolds larger than the cap are reduced to a bounded best-K subset by
# select_manifold_contacts (deepest anchor + greedy position/normal spread),
# which is lossless in practice because the velocity-level rigid solve has
# rank <= 6 per body pair. Single cap, no size bucketing.
RESTITUTION_MANIFOLD_MAX_CONTACTS = 12

_restitution_manifold_max = wp.constant(RESTITUTION_MANIFOLD_MAX_CONTACTS)
_restitution_vecNf = wp.types.vector(length=RESTITUTION_MANIFOLD_MAX_CONTACTS, dtype=wp.float32)
_restitution_vecNi = wp.types.vector(length=RESTITUTION_MANIFOLD_MAX_CONTACTS, dtype=wp.int32)


@wp.kernel
def build_restitution_manifolds(
    body_q_pre_solve: wp.array[wp.transform],
    body_qd_pre_solve: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    body_world: wp.array[wp.int32],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    restitution_contact_active: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_material_restitution: wp.array[float],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    dt: float,
    body_count: int,
    # outputs
    manifold_key: wp.array[wp.int64],
    manifold_head: wp.array[wp.int32],
    manifold_total: wp.array[wp.int32],
    contact_next: wp.array[wp.int32],
    contact_n_K: wp.array[wp.vec4],
    contact_axn_lo_target: wp.array[wp.vec4],
    contact_axn_hi_sigma: wp.array[wp.vec4],
    contact_pos_depth: wp.array[wp.vec4],
):
    """Group contacts that can fire restitution into per-body-pair manifolds.

    The collision pipeline interleaves contacts across pairs, so contacts are
    not pair-contiguous. Each active contact inserts its canonical body-pair
    key (``(lo+1)*(body_count+1) + hi+1 >= 1``; 0 means an empty slot) into an
    open-addressing hash table via ``atomic_cas`` and pushes its contact index
    onto the slot's uncapped linked chain
    (``manifold_head``/``contact_next``); :func:`select_manifold_contacts`
    later reduces each chain to a bounded best-K subset. Fixed-size table, no
    host sync: CUDA-graph-capture safe.

    Alongside, packed per-contact records consumed by
    :func:`solve_manifold_restitution` are cached:

    * ``contact_n_K``: contact normal and effective inverse mass
      ``K = m_inv + (r x n)^T I^-1 (r x n)`` summed over both bodies.
    * ``contact_axn_lo_target``: ``r_lo x n`` (world) and the restitution
      target ``-e * rel_vel_old`` for an impact, the anchored separating
      velocity for an already-separating contact, or zero for a fully
      inelastic impact.
    * ``contact_axn_hi_sigma``: ``r_hi x n`` (world) and the normal sign
      ``sigma`` for the lower-indexed body (-1 when it is the shape0 side),
      with the resting threshold packed into its magnitude.
    * ``contact_pos_depth``: world mid-surface contact point and the
      pre-solve penetration depth (larger = deeper), consumed by the
      best-K selection.

    Everything derives from pre-solve state, so the grouping and records are
    valid for all restitution iterations of the current substep.
    """
    tid = wp.tid()

    if tid >= contact_count[0]:
        return
    if restitution_contact_active[tid] == 0:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1
    mat_nonzero = 0
    restitution = 0.0
    if shape_a >= 0:
        mat_nonzero += 1
        restitution += shape_material_restitution[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        restitution += shape_material_restitution[shape_b]
        body_b = shape_body[shape_b]
    if mat_nonzero > 0:
        restitution /= float(mat_nonzero)
    if body_a == body_b or restitution < 0.0:
        return

    lo = wp.min(body_a, body_b)
    hi = wp.max(body_a, body_b)
    # sigma: the contact's normal sign for the lo body (normal points A -> B)
    sigma = 1.0
    if body_a == lo:
        sigma = -1.0

    X_lo = wp.transform_identity()
    X_hi = wp.transform_identity()
    m_inv_lo = 0.0
    m_inv_hi = 0.0
    I_inv_lo = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_hi = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    com_lo = wp.vec3(0.0)
    com_hi = wp.vec3(0.0)
    qd_lo_pre = wp.spatial_vector()
    qd_hi_pre = wp.spatial_vector()
    gravity_magnitude = 0.0
    if lo >= 0:
        X_lo = body_q_pre_solve[lo]
        m_inv_lo = body_m_inv[lo]
        I_inv_lo = body_I_inv[lo]
        com_lo = body_com[lo]
        qd_lo_pre = body_qd_pre_solve[lo]
        gravity_magnitude = wp.max(gravity_magnitude, wp.length(gravity[body_world[lo]]))
    if hi >= 0:
        X_hi = body_q_pre_solve[hi]
        m_inv_hi = body_m_inv[hi]
        I_inv_hi = body_I_inv[hi]
        com_hi = body_com[hi]
        qd_hi_pre = body_qd_pre_solve[hi]
        gravity_magnitude = wp.max(gravity_magnitude, wp.length(gravity[body_world[hi]]))

    p_lo = contact_point1[tid]
    o_lo = contact_offset1[tid]
    p_hi = contact_point0[tid]
    o_hi = contact_offset0[tid]
    if body_a == lo:
        p_lo = contact_point0[tid]
        o_lo = contact_offset0[tid]
        p_hi = contact_point1[tid]
        o_hi = contact_offset1[tid]

    n = contact_normal[tid]
    r_lo = contact_surface_point(X_lo, p_lo, o_lo) - wp.transform_point(X_lo, com_lo)
    r_hi = contact_surface_point(X_hi, p_hi, o_hi) - wp.transform_point(X_hi, com_hi)
    axn_lo = wp.cross(r_lo, n)
    axn_hi = wp.cross(r_hi, n)

    inv_mass = 0.0
    rel_vel_old = 0.0
    if lo >= 0:
        q_lo = wp.transform_get_rotation(X_lo)
        rxn_lo = wp.quat_rotate_inv(q_lo, axn_lo)
        inv_mass += m_inv_lo + wp.dot(rxn_lo, I_inv_lo * rxn_lo)
        rel_vel_old += sigma * wp.dot(n, velocity_at_point(qd_lo_pre, r_lo))
    if hi >= 0:
        q_hi = wp.transform_get_rotation(X_hi)
        rxn_hi = wp.quat_rotate_inv(q_hi, axn_hi)
        inv_mass += m_inv_hi + wp.dot(rxn_hi, I_inv_hi * rxn_hi)
        rel_vel_old -= sigma * wp.dot(n, velocity_at_point(qd_hi_pre, r_hi))

    if inv_mass == 0.0:
        return
    impact_threshold = 2.0 * gravity_magnitude * dt
    target = 0.0
    if rel_vel_old < -impact_threshold:
        target = -restitution * rel_vel_old
    elif rel_vel_old > impact_threshold:
        target = rel_vel_old

    contact_n_K[tid] = wp.vec4(n[0], n[1], n[2], inv_mass)
    contact_axn_lo_target[tid] = wp.vec4(axn_lo[0], axn_lo[1], axn_lo[2], target)
    contact_axn_hi_sigma[tid] = wp.vec4(axn_hi[0], axn_hi[1], axn_hi[2], sigma * (1.0 + impact_threshold))
    bx_lo = contact_surface_point(X_lo, p_lo, o_lo)
    bx_hi = contact_surface_point(X_hi, p_hi, o_hi)
    p_mid = 0.5 * (bx_lo + bx_hi)
    # pre-solve penetration depth along the A->B normal (larger = deeper)
    depth = sigma * wp.dot(n, bx_hi - bx_lo)
    contact_pos_depth[tid] = wp.vec4(p_mid[0], p_mid[1], p_mid[2], depth)

    key = wp.int64(lo + 1) * wp.int64(body_count + 1) + wp.int64(hi + 1)
    table_size = manifold_key.shape[0]
    slot = wp.int32(key % wp.int64(table_size))
    # linear probing; table capacity >= contact capacity so a slot always exists
    for _attempt in range(table_size):
        prev = wp.atomic_cas(manifold_key, slot, wp.int64(0), key)
        if prev == wp.int64(0) or prev == key:
            wp.atomic_add(manifold_total, slot, 1)
            # heads store index+1 so a zeroed table means empty (memset-cheap reset)
            contact_next[tid] = wp.atomic_exch(manifold_head, slot, tid + 1) - 1
            return
        slot += 1
        if slot == table_size:
            slot = 0


@wp.kernel
def select_manifold_contacts(
    manifold_key: wp.array[wp.int64],
    manifold_head: wp.array[wp.int32],
    manifold_total: wp.array[wp.int32],
    contact_next: wp.array[wp.int32],
    contact_pos_depth: wp.array[wp.vec4],
    contact_n_K: wp.array[wp.vec4],
    # outputs
    manifold_contact: wp.array[wp.int32],
    manifold_size: wp.array[wp.int32],
    contact_sel_score: wp.array[float],
):
    """Reduce each manifold chain to a bounded best-K contact subset.

    One thread per occupied hash slot. The deepest contact anchors the
    selection (ties break to the lower contact index); the remaining picks
    greedily maximize the minimum distance to the already-selected set in a
    combined metric ``|dp|^2 + w |dn|^2`` with ``w`` the squared patch radius
    about the anchor, approximating coverage of the contact Jacobian rows
    (position spread spans the torque rows, normal spread the force rows).
    Farthest-point selection with a per-contact running score
    (``contact_sel_score``; each contact belongs to exactly one manifold, so
    the scratch is race-free): O(K * N) chain walks. Every argmax breaks ties
    to the lower contact index, so the result is independent of chain
    (atomic append) order — deterministic. Manifolds with at most
    ``RESTITUTION_MANIFOLD_MAX_CONTACTS`` contacts keep every contact
    unchanged (fast path). Over-cap manifolds drop exact duplicates in
    position and normal — their max-min score is exactly zero, so they are
    never picked and the selected size can be below the cap. The
    velocity-level rigid solve is rank <= 6 per body pair, so a spread-K
    subset preserves the reachable impulse space.
    """
    tid = wp.tid()

    if manifold_key[tid] == wp.int64(0):
        return
    head = manifold_head[tid] - 1
    if head < 0:
        return

    # fast path: within-cap manifolds keep every contact; chain order is
    # irrelevant because the solve kernel sorts its indices canonically
    total = manifold_total[tid]
    if total <= _restitution_manifold_max:
        n_all = wp.int32(0)
        c = head
        while c >= 0:
            manifold_contact[tid * _restitution_manifold_max + n_all] = c
            n_all += 1
            c = contact_next[c]
        manifold_size[tid] = n_all
        return

    # pass 1: anchor = deepest contact (tie -> lower index)
    anchor = wp.int32(-1)
    best_depth = float(-1.0e30)
    c = head
    while c >= 0:
        depth = contact_pos_depth[c][3]
        if depth > best_depth or (depth == best_depth and (anchor < 0 or c < anchor)):
            best_depth = depth
            anchor = c
        c = contact_next[c]

    if anchor < 0:
        return

    # pass 2: squared patch radius about the anchor -> normal-term weight,
    # and initialize per-contact scores as distance-to-anchor
    d_a = contact_pos_depth[anchor]
    p_anchor = wp.vec3(d_a[0], d_a[1], d_a[2])
    d_an = contact_n_K[anchor]
    n_anchor = wp.vec3(d_an[0], d_an[1], d_an[2])
    r2 = float(0.0)
    c = head
    while c >= 0:
        d_c = contact_pos_depth[c]
        p_c = wp.vec3(d_c[0], d_c[1], d_c[2])
        r2 = wp.max(r2, wp.length_sq(p_c - p_anchor))
        c = contact_next[c]
    w = wp.max(r2, 1.0e-12)
    c = head
    while c >= 0:
        if c == anchor:
            contact_sel_score[c] = -1.0  # selected sentinel
        else:
            d_c = contact_pos_depth[c]
            p_c = wp.vec3(d_c[0], d_c[1], d_c[2])
            d_cn = contact_n_K[c]
            n_c = wp.vec3(d_cn[0], d_cn[1], d_cn[2])
            contact_sel_score[c] = wp.length_sq(p_c - p_anchor) + w * wp.length_sq(n_c - n_anchor)
        c = contact_next[c]

    sel = _restitution_vecNi()
    sel[0] = anchor
    n_sel = wp.int32(1)
    for _pick in range(_restitution_manifold_max - 1):
        # argmax of running min-distance score (tie -> lower index)
        best = wp.int32(-1)
        best_score = float(0.0)
        c = head
        while c >= 0:
            sc = contact_sel_score[c]
            if sc > best_score or (sc == best_score and sc > 0.0 and (best < 0 or c < best)):
                best_score = sc
                best = c
            c = contact_next[c]
        if best < 0:
            break  # nothing left that adds span (or chain exhausted)
        sel[n_sel] = best
        n_sel += 1
        contact_sel_score[best] = -1.0
        # relax remaining scores against the new pick
        d_b = contact_pos_depth[best]
        p_b = wp.vec3(d_b[0], d_b[1], d_b[2])
        d_bn = contact_n_K[best]
        n_b = wp.vec3(d_bn[0], d_bn[1], d_bn[2])
        c = head
        while c >= 0:
            if contact_sel_score[c] > 0.0:
                d_c = contact_pos_depth[c]
                p_c = wp.vec3(d_c[0], d_c[1], d_c[2])
                d_cn = contact_n_K[c]
                n_c = wp.vec3(d_cn[0], d_cn[1], d_cn[2])
                cand = wp.length_sq(p_c - p_b) + w * wp.length_sq(n_c - n_b)
                contact_sel_score[c] = wp.min(contact_sel_score[c], cand)
            c = contact_next[c]

    for j in range(n_sel):
        manifold_contact[tid * _restitution_manifold_max + j] = sel[j]
    manifold_size[tid] = n_sel


@wp.func
def _world_inv_inertia(q: wp.quat, I_inv: wp.mat33) -> wp.mat33:
    R = wp.quat_to_matrix(q)
    return R @ I_inv @ wp.transpose(R)


@wp.kernel
def solve_manifold_restitution(
    body_qd: wp.array[wp.spatial_vector],
    body_q_pre_solve: wp.array[wp.transform],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    manifold_key: wp.array[wp.int64],
    manifold_size: wp.array[wp.int32],
    manifold_contact: wp.array[wp.int32],
    body_count: int,
    contact_n_K: wp.array[wp.vec4],
    contact_axn_lo_target: wp.array[wp.vec4],
    contact_axn_hi_sigma: wp.array[wp.vec4],
    inner_iterations: int,
    outer_iteration: int,
    # outputs
    deltas: wp.array[wp.spatial_vector],
    body_manifold_count: wp.array[float],
):
    """Per-manifold effective-mass restitution solve.

    One thread per occupied hash-table slot (canonical body pair) iterates the
    manifold's contacts Gauss-Seidel style against local working copies of the
    two bodies' velocities, using the per-contact records cached by
    :func:`build_restitution_manifolds` and an accumulated lower-bounded clamp
    on each contact impulse (PGS). On the first outer iteration only
    (``outer_iteration == 0``), a contact whose starting separating velocity
    already exceeds its restitution target gets a bounded negative allowance,
    ``(target - rel_vel_start) / k_eff``, attributing the over-separation to
    the contact's own positional impulse through its effective mass — so the
    velocity pass can reduce a positional over-separation down to the target
    and the authored coefficient is honored across the full ``[0, 1]`` range.
    Later outer iterations clamp at zero: after the first pass, above-target
    separation at a pair can be another manifold's restitution impulse, and a
    re-frozen bound would claw it back (adhesive; dissipates energy at
    ``e = 1``). Contacts starting at or below their target keep the plain
    non-negative clamp on every pass. The one-shot attribution ignores
    off-diagonal effective-mass coupling between contacts, so it bounds the
    allowance but does not strictly guarantee per-contact non-adhesion. The
    manifold's net velocity change is written to ``deltas`` with atomics and
    averaged per body by the caller over manifolds that fired
    (``body_manifold_count``). Restitution targets stay anchored to the
    pre-solve velocities throughout the substep.
    """
    tid = wp.tid()

    key = manifold_key[tid]
    if key == wp.int64(0):
        return

    num_contacts = wp.min(manifold_size[tid], _restitution_manifold_max)

    # gather + sort contact indices so Gauss-Seidel order (and float rounding)
    # is independent of the atomic append order
    con = _restitution_vecNi()
    for k in range(num_contacts):
        con[k] = manifold_contact[tid * _restitution_manifold_max + k]
    for i in range(1, num_contacts):
        v = con[i]
        j = i
        while j > 0:
            if con[j - 1] <= v:
                break
            con[j] = con[j - 1]
            j -= 1
        con[j] = v

    lo = wp.int32(key / wp.int64(body_count + 1)) - 1
    hi = wp.int32(key % wp.int64(body_count + 1)) - 1

    # per-manifold body data; zero response for a static side (lo == -1)
    m_inv_lo = 0.0
    m_inv_hi = 0.0
    I_inv_lo_world = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_hi_world = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v_lo0 = wp.vec3(0.0)
    w_lo0 = wp.vec3(0.0)
    v_hi0 = wp.vec3(0.0)
    w_hi0 = wp.vec3(0.0)
    if lo >= 0:
        m_inv_lo = body_m_inv[lo]
        I_inv_lo_world = _world_inv_inertia(wp.transform_get_rotation(body_q_pre_solve[lo]), body_I_inv[lo])
        qd = body_qd[lo]
        v_lo0 = wp.spatial_top(qd)
        w_lo0 = wp.spatial_bottom(qd)
    if hi >= 0:
        m_inv_hi = body_m_inv[hi]
        I_inv_hi_world = _world_inv_inertia(wp.transform_get_rotation(body_q_pre_solve[hi]), body_I_inv[hi])
        qd = body_qd[hi]
        v_hi0 = wp.spatial_top(qd)
        w_hi0 = wp.spatial_bottom(qd)
    v_lo = v_lo0
    w_lo = w_lo0
    v_hi = v_hi0
    w_hi = w_hi0

    # Per-contact impulse lower bound, frozen at the sweep-start working
    # velocities of the FIRST outer iteration only: the positional solve can
    # reconstruct a separating velocity above the restitution target, and the
    # velocity pass must be able to pull that overshoot back down (Eq. 34
    # drives to the target from either side). Bounding the accumulator at
    # (target - rel_vel_start) / k_eff instead of zero allows that reduction
    # and no more. Later outer iterations must NOT re-freeze the bound:
    # above-target separation there can be another manifold's restitution
    # impulse from an earlier pass, and clawing it back is adhesive (measured
    # 23% KE loss at e=1 for a body impacting two supports simultaneously).
    lambda_min = _restitution_vecNf()
    if outer_iteration == 0:
        for k in range(num_contacts):
            c = con[k]
            d_nK = contact_n_K[c]
            k_eff = d_nK[3]
            if k_eff == 0.0:
                continue
            n = wp.vec3(d_nK[0], d_nK[1], d_nK[2])
            d_lot = contact_axn_lo_target[c]
            axn_lo = wp.vec3(d_lot[0], d_lot[1], d_lot[2])
            d_his = contact_axn_hi_sigma[c]
            axn_hi = wp.vec3(d_his[0], d_his[1], d_his[2])
            sigma = wp.sign(d_his[3])
            rel_vel_start = sigma * (wp.dot(n, v_lo) + wp.dot(axn_lo, w_lo) - wp.dot(n, v_hi) - wp.dot(axn_hi, w_hi))
            lambda_min[k] = wp.min((d_lot[3] - rel_vel_start) / k_eff, 0.0)

    # Gauss-Seidel sweeps with accumulated clamp on local working copies
    lambda_acc = _restitution_vecNf()
    fired = wp.bool(False)
    for _it in range(inner_iterations):
        for k in range(num_contacts):
            c = con[k]
            d_nK = contact_n_K[c]
            k_eff = d_nK[3]
            if k_eff == 0.0:
                continue
            n = wp.vec3(d_nK[0], d_nK[1], d_nK[2])
            d_lot = contact_axn_lo_target[c]
            axn_lo = wp.vec3(d_lot[0], d_lot[1], d_lot[2])
            target = d_lot[3]
            d_his = contact_axn_hi_sigma[c]
            axn_hi = wp.vec3(d_his[0], d_his[1], d_his[2])
            sigma_threshold = d_his[3]
            sigma = wp.sign(sigma_threshold)
            impact_threshold = wp.abs(sigma_threshold) - 1.0

            rel_vel_new = sigma * (wp.dot(n, v_lo) + wp.dot(axn_lo, w_lo) - wp.dot(n, v_hi) - wp.dot(axn_hi, w_hi))
            if target == 0.0 and wp.abs(rel_vel_new) <= impact_threshold:
                continue
            dv = (target - rel_vel_new) / k_eff
            new_acc = wp.max(lambda_acc[k] + dv, lambda_min[k])
            d_lambda = new_acc - lambda_acc[k]
            lambda_acc[k] = new_acc
            if d_lambda == 0.0:
                continue
            fired = True
            s = sigma * d_lambda
            v_lo += n * (m_inv_lo * s)
            w_lo += (I_inv_lo_world * axn_lo) * s
            v_hi -= n * (m_inv_hi * s)
            w_hi -= (I_inv_hi_world * axn_hi) * s

    if not fired:
        return

    # net manifold contribution; cross-manifold coupling is Jacobi (averaged
    # per body by the caller over manifolds that fired)
    if lo >= 0 and (m_inv_lo > 0.0 or wp.ddot(I_inv_lo_world, I_inv_lo_world) > 0.0):
        wp.atomic_add(deltas, lo, wp.spatial_vector(v_lo - v_lo0, w_lo - w_lo0))
        wp.atomic_add(body_manifold_count, lo, 1.0)
    if hi >= 0 and (m_inv_hi > 0.0 or wp.ddot(I_inv_hi_world, I_inv_hi_world) > 0.0):
        wp.atomic_add(deltas, hi, wp.spatial_vector(v_hi - v_hi0, w_hi - w_hi0))
        wp.atomic_add(body_manifold_count, hi, 1.0)


@wp.kernel
def apply_restitution_deltas(
    deltas: wp.array[wp.spatial_vector],
    body_manifold_count: wp.array[float],
    qd_out: wp.array[wp.spatial_vector],
):
    """Apply per-body restitution deltas averaged over fired manifolds.

    Consumes and clears both accumulators so the next restitution iteration
    starts from zero without extra zeroing launches (in-place path only; the
    ``requires_grad`` path uses fresh buffers instead).
    """
    tid = wp.tid()
    inv_weight = body_manifold_count[tid]
    if inv_weight <= 0.0:
        return
    qd_out[tid] = qd_out[tid] + deltas[tid] / inv_weight
    deltas[tid] = wp.spatial_vector()
    body_manifold_count[tid] = 0.0
