# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-pair contact reduction: post-narrow-phase compaction of redundant contacts.

Multi-shape bodies multiply narrow-phase output: a foot approximated by 7
cylinders emits up to 28 candidate contacts against a plane and up to 49
against another such foot, even when the underlying physics is one flat patch.
The pass represents that patch with one depth slot and two families of six
sampled footprint slots rather than treating collider decomposition as
physical detail.

**Approximation, stated precisely.** For contacts RETAINED at the true hull, an
interior point's normal force is a convex combination of hull-point normal
forces (support/tipping statics preserved) and translational friction capacity
``sum mu*N_i`` is position-independent.  But the hull itself is SAMPLED by
:data:`BODY_PAIR_NUM_DIRECTIONS` fixed scan directions, and the sampling error
carries NO universal bound -- the following are EXAMPLES, not ceilings.  For a
continuous circular footprint whose radius is represented at every scan
direction, the retained regular polygon has a radial deficit of
``1 - cos(pi/N)`` halfway between directions (~13% at six directions).  That
regular-circle construction is not a guarantee for an arbitrary finite or
nominally dense candidate set.  A sparse hull can lose the entire support of an
unsampled direction: candidates at -60/120 degrees plus a true hull vertex at
30 degrees that trails each of the six scan projections by an epsilon lose ALL
of the 30-degree support (~58% of the patch radius there, and the discarded
point is a true hull vertex).  Support, tipping,
and translational friction are therefore preserved only to scene-dependent
tolerances pinned by the test suite -- not exactly, and not within any
shape-independent percentage.  Row count and placement also affect
iterative-solver compliance, so equal capacity does not imply identical
realized load distribution.

This pass runs after the narrow phase has written the ``Contacts`` buffer and
compacts it in two kernels over the live contact range -- register, then select:

* contacts are grouped by ``(group0, group1, normal bin, spatial cell)`` where
  the group is the body (or the shape itself for static geometry, so distinct
  static colliders never merge), the bin is a face of this module's own
  Z-aligned icosahedron (see :data:`BP_FACE_NORMALS` for why it is not the
  shared table), and the cell is the contact's position on that face plane
  quantized relative to the pair's own reference body (see
  :func:`_contact_group_key`);
* every contact enters its group's depth slot and all
  :data:`BODY_PAIR_NUM_DIRECTIONS` spatial-extreme slots, competing on
  projection alone, and touching contacts (separation ``<= 0``) also enter a
  second family of extreme slots closed to speculative ones (see
  :data:`TOUCHING_SLOT_OFFSET`); each slot holds its instantaneous winner (see
  :data:`BODY_PAIR_REDUCTION_SLOTS` for the policies this replaced);
* each contact then checks whether any value it submitted won its slot, and the
  survivors are compacted in place so ``rigid_contact_count`` itself drops.
  Everything else is discarded -- typically interior points whose force is a
  convex combination of the hull points', though a sparsely populated hull can
  also lose true vertices that fall between scan directions (see above).

**When this pays.** Only bodies whose collision is decomposed into several
primitives generate more candidates per pair than the slot budget. Dense
multi-collider patches can repay the extra registration and compaction work by
substantially reducing solver rows; scenes dominated by one small manifold per
body pair cannot and pay pure overhead. It is opt-in for that reason.
``stats()`` reports paired input/kept totals and an ``identity_frames`` counter;
frequent identity frames or a kept/input ratio near one mean this scene does not
want it. Benchmark the complete collision-plus-solver step on the target GPU
rather than inferring speedup from contact count alone.

Depth is ranked with the canonical signed effective-surface separation
``dot(normal, point1 - point0) - margin0 - margin1``.  It picks the depth
representative and its sign gates the touching slot family; it never biases
the ordering inside a spatial slot.  The pass still has an explicit
spatial-cell length scale, and no depth classification can leave a patch
without footprint support.

Reduction never crosses a normal bin, so multi-patch configurations (a body
touching floor and wall at once) keep representatives of every patch.  On
hashtable probe exhaustion the pass fails open: a contact that cannot be
registered within the bounded probe budget is kept, never dropped.  Probe
exhaustion can mean a full table or a long collision cluster; telemetry does
not claim to distinguish the two.

**Memoryless selection.** The kept set is a function of the current contact
geometry only.  On curved or mesh support (flat analytic planes are immune) the
score differences between a patch's near-symmetric contacts are curvature-sized,
so the winners can hand off between steps.

**XPBD is not a supported consumer.** XPBD predicts body poses before deciding
which speculative candidates are active, while this pass selects footprint
representatives from the pre-prediction contact geometry. A spatial winner can
therefore remain speculative after prediction while a discarded interior
candidate becomes penetrating. Increasing XPBD iterations reduced a curved-foot
rocking residual in one characterization scene but did not establish a
class-wide selection invariant, so XPBD rejects reduced buffers. Steel is
the supported solver.

**Known characteristic -- torsional patch friction.** During twist, friction at
every contact is saturated at ``mu*N_i``, so the resisting torque is
``mu * sum(N_i * r_i)`` -- fixed by WHERE the normal load sits, with no
remaining freedom for the solver to redistribute. Keeping only rim extremes
forces the load toward the largest lever arms, increasing that sum. For a
uniform circular patch, concentrating all load at the rim produces 1.5 times
the continuum uniform-pressure torsion; other footprint shapes have different
ratios, so this is not a universal ceiling. A dense point set approximates the
uniform-pressure integral only by accident of even loading -- nobody reweights
in either case -- and no placement of boundary points can match both
translational capacity (``sum mu*N_i``) and the continuum torsion
(``2/3 mu*W*R`` for a circle) simultaneously. The bias is inherent to a
reduction that keeps a patch's boundary points. The principled fix, if yaw
transfer measurably suffers, is an explicit per-patch torsion row in the
solver, not point placement. Watch yaw-tracking error when validating policies
whose feet pivot.

All buffers are allocated at construction and all launches are fixed-size, so
the pass is CUDA-graph-capture compatible.  Its scratch is per-frame only, like
the rest of the pipeline's scratch, so graphs that share one pipeline must not
replay concurrently.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..sim.contacts import contact_surface_separation
from ..sim.model import Model
from .contact_reduction import float_flip
from .hashtable import (
    HASHTABLE_EMPTY_KEY,
    HashTable,
    _hashtable_hash,
)

# Normal-binning table PRIVATE to body-pair reduction: the icosahedron from
# :mod:`contact_reduction`, rotated so a face CENTER points at world +Z and the
# yaw chosen to maximize the minimum bin margin at +/-X and +/-Y.
#
# The shared table stores the icosahedron Y-up, which puts +Z on the boundary
# between two equatorial faces to within 7e-8 of dot product. A flat plane is
# immune (its normal is bit-identical every step, so the argmax is stable), but
# on any curved or mesh surface in a Z-up world the patch normals wobble across
# that boundary, contacts regroup between bins step to step, and the kept set
# churns: measured 20x angular-velocity noise and a permanent rocking limit
# cycle that never settles. With this table the margin at +/-Z is 0.2546 -- a
# ground normal must tilt more than 20 degrees to change bins -- and 0.066 at
# +/-X/+/-Y (walls, ~4 degrees). Boundaries must exist somewhere on the sphere;
# they just must not pass through the one direction every scene is built
# around. Private copy on purpose: the mesh reducer consumes the shared table,
# and re-orienting that path is out of scope (same reasoning as
# BODY_PAIR_NUM_DIRECTIONS).
#
# Derivation (offline, float64): R0 aligns shared-table face 10 with +Z; yaw
# 55.75 deg about Z then maximizes min(margin(+/-X), margin(+/-Y)); rows
# renormalized. The two axis-aligned rows are written exactly.
# fmt: off
_BP_FACE_NORMALS_DATA = (
    -0.57095543,  0.75026579,  0.33333337,
    -0.64404845,  0.17218052,  0.74535593,
    -0.86918069, -0.36525859,  0.33333330,
    -0.93522697, -0.11932883, -0.33333339,
    -0.75091368,  0.57010308, -0.33333336,
     0.36427171,  0.86959472, -0.33333337,
    -0.17291149,  0.64385240, -0.74535609,
     0.47113684,  0.47167205,  0.74535599,
     0.11826704,  0.93536185,  0.33333335,
     0.17291155, -0.64385245,  0.74535603,
     0.0,         0.0,         1.0,
    -0.11826707, -0.93536184, -0.33333335,
    -0.36427169, -0.86959473,  0.33333337,
     0.0,         0.0,        -1.0,
    -0.47113677, -0.47167196, -0.74535609,
     0.93522695,  0.11932898,  0.33333339,
     0.75091367, -0.57010306,  0.33333340,
     0.57095545, -0.75026579, -0.33333334,
     0.64404840, -0.17218051, -0.74535597,
     0.86918063,  0.36525872, -0.33333332,
)
# fmt: on
BP_NUM_NORMAL_BINS = 20
_bp_face_normals_mat = wp.types.matrix(shape=(BP_NUM_NORMAL_BINS, 3), dtype=wp.float32)
BP_FACE_NORMALS = _bp_face_normals_mat(*_BP_FACE_NORMALS_DATA)


def _up_axis_rotation(up_axis: int) -> tuple:
    """Exact rotation (rows of a permutation matrix) mapping ``up_axis`` to +Z.

    The bin table is oriented so its widest margins protect the WORLD UP
    direction; Newton models can be X-, Y-, or Z-up, so grouping geometry is
    rotated into a Z-up frame first.  The entries are exactly 0/1 (cyclic axis
    permutations, det +1), so the rotation introduces no rounding.
    """
    if up_axis == 0:  # X-up: X->Z, Y->X, Z->Y
        return ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    if up_axis == 1:  # Y-up: X->Y, Y->Z, Z->X
        return ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# Solver-visible contact material fields. Shapes on one body may only share a
# reduction group when ALL of these are exactly equal: the surviving contacts'
# shape ids are what the solvers read material laws from, so merging shapes
# with different materials could delete e.g. a high-friction pad in favor of
# the surrounding low-friction colliders and change grip. Exact equality, not
# a hash -- a hash collision would silently merge incompatible laws.
_MATERIAL_FIELDS = (
    "shape_material_ke",
    "shape_material_kd",
    "shape_material_kf",
    "shape_material_ka",
    "shape_material_kh",
    "shape_material_mu",
    "shape_material_mu_torsional",
    "shape_material_mu_rolling",
    "shape_material_restitution",
)


def shape_material_attribute_names(model) -> tuple[str, ...]:
    """Return every per-shape custom attribute the model registers, by full name."""
    return tuple(
        sorted(
            name
            for name, spec in model._iter_attribute_specs()
            if spec.frequency == Model.AttributeFrequency.SHAPE and spec.assignment == Model.AttributeAssignment.MODEL
        )
    )


def build_reduction_groups(model, material_attributes: tuple[str, ...] | None = None) -> tuple:
    """Assign each shape a reduction-group id (host side, at construction).

    Static shapes keep unique ids so distinct static colliders never merge.
    Dynamic shapes share a group only per (body, exact material signature): the
    ``shape_material_*`` fields plus the named per-shape custom attributes
    (every one the model registers when ``material_attributes`` is ``None``).

    Returns:
        ``(ids, group_count)`` where ids is an int32 array of length
        ``shape_count``.
    """
    n = int(model.shape_count)
    shape_body = model.shape_body.numpy()
    mats = []
    for name in _MATERIAL_FIELDS:
        arr = getattr(model, name, None)
        if arr is not None:
            mats.append(arr.numpy())
    registered = shape_material_attribute_names(model)
    if material_attributes is None:
        material_attributes = registered
    for full_name in material_attributes:
        if full_name not in registered:
            raise ValueError(f"material attribute {full_name!r} is not a per-shape custom attribute of this model")
        namespace, _, name = full_name.rpartition(":")
        values = getattr(getattr(model, namespace) if namespace else model, name)
        mats.append(values.numpy() if isinstance(values, wp.array) else np.asarray(values, dtype=object))
    ids = np.zeros(n, dtype=np.int32)
    class_ids: dict = {}
    next_id = 0
    for shape in range(n):
        body = int(shape_body[shape])
        if body < 0:
            key = ("static", shape)
        else:
            key = ("body", body, tuple(tuple(np.ravel(m[shape]).tolist()) for m in mats))
        gid = class_ids.get(key)
        if gid is None:
            gid = next_id
            next_id += 1
            class_ids[key] = gid
        ids[shape] = gid
    return ids, next_id


def build_reduction_group_pair_bound(pairs: np.ndarray, shape_group) -> int:
    """Count the distinct ``(group_a, group_b)`` pairs that ``pairs`` can produce.

    The reduction key is ``(group_a, group_b, normal bin, spatial cell)``.  This
    returns an EXACT upper bound on the pair dimension alone: the image of the
    candidate shape pairs under :func:`build_reduction_groups`, deduped the same
    unordered way :func:`_make_group_key` packs it.  The caller passes the
    explicit pair list in explicit broad-phase mode and the model's default
    pairs otherwise, which SAP and NXN filter the same way.

    It bounds ONE dimension of the key, not the entry count, and is loose in
    both directions: a pair spread over several normal bins or spatial cells
    occupies several entries, while pairs that are reachable but never
    simultaneously in contact occupy none.  How loose is scene-dependent --
    scenes whose candidate pairs are mostly potential rather than active sit
    far below it.  It is therefore a sizing ANCHOR, not a guarantee: its value
    is that it scales with the scene topology the group count is actually a
    function of, rather than with contact-buffer capacity, which it is not.
    Read ``stats()["max_hashtable_entries"]`` for what a given scene needs.

    Args:
        pairs: Host candidate shape pairs, shape ``(n, 2)``.
        shape_group: Per-shape group ids from :func:`build_reduction_groups`.

    Returns:
        The number of distinct unordered group pairs.
    """
    p = np.asarray(pairs).reshape(-1, 2)
    if p.size == 0:
        return 0
    sg = np.asarray(shape_group, dtype=np.int64)
    ga, gb = sg[p[:, 0]], sg[p[:, 1]]
    lo, hi = np.minimum(ga, gb), np.maximum(ga, gb)
    # Group ids are asserted below MAX_GROUP_ID (2**23) at pipeline construction,
    # so this packing stays far inside int64 and cannot alias two distinct pairs.
    return int(np.unique(lo * (int(sg.max()) + 1) + hi).size)


@wp.func
def _bp_get_slot(normal: wp.vec3) -> int:
    """Return the body-pair bin whose face normal best matches ``normal``.

    Full linear scan over the rotated table; the shared reducer's Y-cap pruning
    does not apply to a rotated orientation, and 20 dot products are noise next
    to the hashtable probe this feeds.
    """
    best_slot = int(0)
    max_dot = wp.dot(normal, wp.vec3(BP_FACE_NORMALS[0, 0], BP_FACE_NORMALS[0, 1], BP_FACE_NORMALS[0, 2]))
    for i in range(1, wp.static(BP_NUM_NORMAL_BINS)):
        d = wp.dot(normal, wp.vec3(BP_FACE_NORMALS[i, 0], BP_FACE_NORMALS[i, 1], BP_FACE_NORMALS[i, 2]))
        if d > max_dot:
            max_dot = d
            best_slot = i
    return best_slot


@wp.func
def _bp_project_point_to_plane(bin_normal_idx: int, point: wp.vec3) -> wp.vec2:
    """Project ``point`` onto the local 2D frame of a body-pair bin face."""
    face_normal = wp.vec3(
        BP_FACE_NORMALS[bin_normal_idx, 0],
        BP_FACE_NORMALS[bin_normal_idx, 1],
        BP_FACE_NORMALS[bin_normal_idx, 2],
    )
    if wp.abs(face_normal[1]) < 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(1.0, 0.0, 0.0)
    u = wp.normalize(ref - wp.dot(ref, face_normal) * face_normal)
    v = wp.cross(face_normal, u)
    return wp.vec2(wp.dot(point, u), wp.dot(point, v))


# Scan directions used to pick footprint extremes on a normal bin's face plane.
# Deliberately independent of the mesh reducer's own direction count so tuning
# that path cannot silently change body-pair behaviour.
#
# Six is a measured floor, not a value carried over. Each extra direction costs a
# slot, an atomic per contact, and a kept contact whose row the solver carries,
# so fewer was tried first: on the randomized primitive piles of the property
# test, peak body speed against the unreduced run's 95 m/s comes out at 95 with
# six directions, 199 with four, and diverges outright with five. Fewer
# directions under-sample a patch's hull and leave a face rocking on too few
# support points.
BODY_PAIR_NUM_DIRECTIONS = 6


@wp.func
def _direction_2d(dir_idx: int) -> wp.vec2:
    """Unit 2D direction at ``dir_idx * 2pi / BODY_PAIR_NUM_DIRECTIONS``."""
    angle = float(dir_idx) * (2.0 * wp.pi / float(wp.static(BODY_PAIR_NUM_DIRECTIONS)))
    return wp.vec2(wp.cos(angle), wp.sin(angle))


# Value slots per (body pair, normal bin, spatial cell) entry: one spatial
# extreme per scan direction over all contacts, the group's deepest contact,
# and one extreme per scan direction over touching contacts.
#
# Gating the all-contact family on a depth window starves any patch whose gap
# spread exceeds the window: a tilted box face keeps only its deepest corner,
# pivots on it and diverges. The touching family is a separate set of slots
# instead, because on a compound body the all-contact extremes can all be
# speculative (a tumbler's wall hulls hovering inside the gap sit further out
# than its base), leaving one load-bearing contact and a rocking body.
DEEPEST_SLOT = BODY_PAIR_NUM_DIRECTIONS
TOUCHING_SLOT_OFFSET = BODY_PAIR_NUM_DIRECTIONS + 1
BODY_PAIR_REDUCTION_SLOTS = 2 * BODY_PAIR_NUM_DIRECTIONS + 1


# Bit budget of the 63-bit group key. Group ids are asserted against this at
# pipeline construction: aliasing two groups could evict a patch's deepest
# contact, which would break the strict keep-deepest guarantee. Replicated
# scenes can create one material-equivalence group per body per world, so 21
# bits can be exhausted before the contact or solver buffers are. Two bits are
# borrowed from each spatial coordinate to support larger replicated scenes
# without adding another lookup to the contact hot path.
GROUP_ID_BITS = 23
MAX_GROUP_ID = (1 << GROUP_ID_BITS) - 1
# Cell coordinates are packed EXACTLY as two signed 6-bit values, measured from
# the pair's own reference body (see _contact_group_key), so the range covers a
# +/-31-cell span from that reference -- 7.75 m at the default cell size, or
# 15.5 m across the full packed span. Beyond it the coordinate clamps
# to the border cell, which only ever over-competes (the deepest of the merged
# region is still kept) and is counted in the clamp telemetry.
CELL_COORD_BITS = 6
CELL_COORD_MAX = (1 << (CELL_COORD_BITS - 1)) - 1


@wp.func
def _make_group_key(group_a: int, group_b: int, bin_id: int, cx: int, cy: int) -> wp.uint64:
    """Pack (group_a, group_b, normal bin, exact spatial cell) into 63 bits.

    Layout: ``[62:40] group_a (23b) | [39:17] group_b (23b) | [16:12] bin (5b)
    | [11:6] cx (6b) | [5:0] cy (6b)``.  All fields are exact within their
    asserted/clamped ranges, so two distinct groups can never alias.
    """
    ux = wp.uint64(wp.clamp(cx, -CELL_COORD_MAX, CELL_COORD_MAX) + CELL_COORD_MAX)
    uy = wp.uint64(wp.clamp(cy, -CELL_COORD_MAX, CELL_COORD_MAX) + CELL_COORD_MAX)
    return (
        (wp.uint64(group_a) << wp.uint64(40))
        | (wp.uint64(group_b) << wp.uint64(17))
        | ((wp.uint64(bin_id) & wp.uint64(0x1F)) << wp.uint64(12))
        | (ux << wp.uint64(6))
        | uy
    )


# Telemetry slots in the reducer's int32 stats array.  The three watermarks are
# whole-run maxima.  Event slots are per-frame atomic scratch: the preparation
# kernel clears them, producers increment them, and the single-thread count
# writer folds them into int64 lifetime totals.  This keeps high-contention
# atomics at their fast/native width without allowing a long training run to
# wrap its public counters.
STAT_VIOLATIONS = 0
STAT_FAIL_OPEN = 1
STAT_OUTRANKED = 2
STAT_ENTRY_WATERMARK = 3
STAT_CELL_CLAMPS = 4
STAT_CONTACTS_IN = 5
STAT_CONTACTS_KEPT = 6
STAT_INPUT_OVERFLOWS = 7
STAT_FALLBACK_FRAMES = 8
STAT_IDENTITY_FRAMES = 9
STAT_COUNT = 10

# 64-bit whole-run accumulators, kept in a separate array.  Contact sums and
# even nominally rare events can outgrow int32 in long-running training jobs.
STAT64_TOTAL_FRAMES = 0
STAT64_SUM_CONTACTS_IN = 1
STAT64_SUM_CONTACTS_KEPT = 2
STAT64_PROBE_FAILURES = 3
STAT64_CELL_CLAMP_EVENTS = 4
STAT64_INVARIANT_VIOLATIONS = 5
STAT64_OUTRANKED_DISCARDS = 6
STAT64_INPUT_OVERFLOW_FRAMES = 7
STAT64_FALLBACK_FRAMES = 8
STAT64_IDENTITY_FRAMES = 9
STAT64_COUNT = 10

# Per-frame device state, reset by the preparation kernel each reduce.
FRAME_WORK_COUNT = 0  # validated loop bound for every reducer kernel
FRAME_INPUT_OVERFLOW = 1  # narrow phase emitted more contacts than capacity
FRAME_FALLBACK = 2  # bounded group-key probe failed: keep every contact this frame
FRAME_IDENTITY = 3  # every group holds one contact: reduction is a no-op
FRAME_STATE_COUNT = 4

# Linear-probe budget for the reduction's hashtable accesses. The table is
# provisioned for load far below saturation (measured 0.08 on the reference
# scene), where expected probe chains are a handful of slots; a chain this long
# means the table is effectively saturated for this frame and the pass must
# fail open as a whole rather than let CUDA scheduling decide which groups get
# the last entries (a scheduling-dependent kept set) or let probes walk the
# entire capacity (a quadratic-cost cliff).
PROBE_BUDGET = 128


@wp.func
def _bp_find_or_insert(
    key: wp.uint64,
    keys: wp.array[wp.uint64],
    active_slots: wp.array[wp.int32],
) -> int:
    """``hashtable_find_or_insert`` with the probe budget applied."""
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    for _i in range(wp.static(PROBE_BUDGET)):
        stored_key = keys[idx]
        if stored_key == key:
            return idx
        if stored_key == HASHTABLE_EMPTY_KEY:
            old_key = wp.atomic_cas(keys, idx, HASHTABLE_EMPTY_KEY, key)
            if old_key == HASHTABLE_EMPTY_KEY:
                active_idx = wp.atomic_add(active_slots, capacity, 1)
                if active_idx < capacity:
                    active_slots[active_idx] = idx
                return idx
            elif old_key == key:
                return idx
        idx = (idx + 1) & capacity_mask
    return -1


@wp.func
def _bp_find(key: wp.uint64, keys: wp.array[wp.uint64]) -> int:
    """``hashtable_find`` with the probe budget applied (read-only lookup)."""
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    for _i in range(wp.static(PROBE_BUDGET)):
        stored_key = keys[idx]
        if stored_key == key:
            return idx
        if stored_key == HASHTABLE_EMPTY_KEY:
            return -1
        idx = (idx + 1) & capacity_mask
    return -1


@wp.kernel(enable_backward=False)
def _prepare_frame_kernel(
    contact_count: wp.array[wp.int32],
    reducer_capacity: int,
    # outputs
    frame_state: wp.array[wp.int32],
    stats: wp.array[wp.int32],
):
    """Derive this frame's validated work count and reset the per-frame flags.

    Newton's narrow phase reserves contact indices with an atomic add BEFORE
    checking capacity, so ``rigid_contact_count`` can legitimately exceed
    ``rigid_contact_max`` on overflow (the excess batches are simply not
    written).  The raw counter is therefore NOT a safe array bound.  Every
    reducer kernel loops over the work count derived here instead: on overflow
    it is zero, the whole pass becomes a no-op, and the counter and the
    materialized contact prefix reach the solver exactly as an unreduced
    pipeline would deliver them -- preserving both memory safety and the
    engine's overflow diagnostics.  Reducing a clamped prefix instead would be
    wrong: batched reservations can leave unwritten holes where a batch
    crossed capacity.
    """
    # Per-frame event scratch.  Watermark slots deliberately remain intact.
    stats[wp.static(STAT_VIOLATIONS)] = 0
    stats[wp.static(STAT_FAIL_OPEN)] = 0
    stats[wp.static(STAT_OUTRANKED)] = 0
    stats[wp.static(STAT_CELL_CLAMPS)] = 0
    stats[wp.static(STAT_INPUT_OVERFLOWS)] = 0
    stats[wp.static(STAT_FALLBACK_FRAMES)] = 0
    stats[wp.static(STAT_IDENTITY_FRAMES)] = 0

    raw = contact_count[0]
    if raw >= 0 and raw <= reducer_capacity:
        frame_state[wp.static(FRAME_WORK_COUNT)] = raw
        frame_state[wp.static(FRAME_INPUT_OVERFLOW)] = 0
    else:
        frame_state[wp.static(FRAME_WORK_COUNT)] = 0
        frame_state[wp.static(FRAME_INPUT_OVERFLOW)] = 1
        stats[wp.static(STAT_INPUT_OVERFLOWS)] = 1
    frame_state[wp.static(FRAME_FALLBACK)] = 0
    frame_state[wp.static(FRAME_IDENTITY)] = 0


@wp.func
def _contact_group_key(
    i: int,
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_group: wp.array[wp.int32],
    up_rotation: wp.mat33,
    cell_size: float,
):
    """Compute (key, gap, face-plane position, cell_clamped) for contact ``i``.

    The gap is ``dot(normal, point1 - point0) - margin0 - margin1`` -- the one
    canonical signed-separation convention the solvers consume (normal points
    shape0 -> shape1, positive = gap). Depth must be ranked with exactly that
    formula: a hand-rolled variant with the opposite point order ranked
    SHALLOWEST as deepest and made the pass keep hovering candidates while
    discarding the load-bearing contacts.
    Group ids come from :func:`build_reduction_groups`: static shapes keep
    unique ids, and dynamic shapes share a group per (body, exact material
    signature), so contacts carrying different solver-visible material laws
    never compete for the same slots.  The pair is unordered but the stored
    normal is directed shape0 -> shape1, and the narrow phase orders endpoints
    by geometry type -- one physical patch between two mixed-collider bodies
    can arrive with BOTH orientations.  Binning therefore uses a canonical
    group normal (negated when the group endpoints are swapped) while the gap
    keeps the original directed convention.  Grouping geometry is additionally
    rotated so the model's up axis maps to +Z, where the private bin table has
    its widest margins.  The key also carries a spatial cell -- the contact's
    position quantized on the bin's face plane by ``cell_size`` -- so multiple
    same-normal patches far apart on ONE shape pair (a long body across two
    regions of a terrain collider) each get their own deepest + extremes
    instead of competing for a single slot set.

    Both the cell and the returned face-plane position are measured from the
    pair's OWN reference body, not the world origin, because
    :func:`project_point_to_plane` is a pure linear map and the cell field is
    only 6 bits per axis: absolute coordinates pin every contact past ~8 m to
    the border cell, which silently disables the subdivision for all but the
    envs nearest the origin.  Subtracting a per-group constant cannot change
    which contact is extreme in a direction, so the shift is free, and it also
    recovers the float32 precision that large world offsets spend on the
    exponent.
    """
    s0 = contact_shape0[i]
    s1 = contact_shape1[i]

    b0 = shape_body[s0]
    b1 = shape_body[s1]

    p0_w = contact_point0[i]
    if b0 >= 0:
        p0_w = wp.transform_point(body_q[b0], p0_w)
    p1_w = contact_point1[i]
    if b1 >= 0:
        p1_w = wp.transform_point(body_q[b1], p1_w)

    n = contact_normal[i]
    gap = contact_surface_separation(p0_w, p1_w, n, contact_margin0[i], contact_margin1[i])
    # Group and rank the physical contact footprint, not the raw support-shape
    # witnesses.  The solver-visible effective surfaces are p0 + n*margin0 and
    # p1 - n*margin1, so unequal effective radii/margins move their midpoint.
    # Using the raw midpoint can split equivalent surface contacts into
    # different cells (or change their extrema) whenever n differs from the
    # selected normal-bin face.
    surface0_w = p0_w + n * contact_margin0[i]
    surface1_w = p1_w - n * contact_margin1[i]
    center = 0.5 * (surface0_w + surface1_w)

    g0 = shape_group[s0]
    g1 = shape_group[s1]
    ga = wp.min(g0, g1)
    gb = wp.max(g0, g1)
    # Canonical group normal: contacts of one patch arriving with swapped
    # endpoints carry opposite directed normals; without this they land in
    # opposite bins and never compete, keeping two slot sets for one patch.
    n_group = n
    if g1 < g0:
        n_group = -n

    # Cell origin: the pair's lowest-indexed dynamic body, or the world origin
    # when both sides are static. Chosen from body indices rather than from the
    # contact's own shape order so the origin is the same for every contact of
    # the group -- otherwise contacts of one patch would land in different cells.
    ref = wp.vec3(0.0, 0.0, 0.0)
    if b0 >= 0 and (b1 < 0 or b0 <= b1):
        ref = wp.transform_get_translation(body_q[b0])
    elif b1 >= 0:
        ref = wp.transform_get_translation(body_q[b1])

    bin_id = _bp_get_slot(up_rotation * n_group)
    pos_2d = _bp_project_point_to_plane(bin_id, up_rotation * (center - ref))
    # Round to nearest, not floor: that puts the reference body's origin at the
    # CENTER of cell 0 rather than on its corner. Flooring relative coordinates
    # would place the origin exactly on a cell boundary, so a patch centered
    # under the body -- the common case, e.g. a foot's colliders -- would always
    # straddle up to four cells and get four slot sets instead of one.
    # Clamp in FLOAT before converting: an extreme finite coordinate would
    # overflow the int32 conversion itself and corrupt both the grouping and
    # the clamp telemetry. The clamp bound is one cell beyond the packed range
    # so the out-of-range condition stays observable after conversion.
    fx = wp.floor(pos_2d[0] / cell_size + 0.5)
    fy = wp.floor(pos_2d[1] / cell_size + 0.5)
    bound = float(CELL_COORD_MAX + 1)
    cx = wp.int32(wp.clamp(fx, -bound, bound))
    cy = wp.int32(wp.clamp(fy, -bound, bound))
    key = _make_group_key(ga, gb, bin_id, cx, cy)
    # Report the clamp rather than hide it: clamped cells merge distant regions
    # of one shape pair, which only over-competes, but a sustained nonzero count
    # means one body pair spans more than +/-31 cells and reduction quality
    # across it is no longer what the cell size promises.
    clamped = wp.abs(cx) > CELL_COORD_MAX or wp.abs(cy) > CELL_COORD_MAX
    return key, gap, pos_2d, clamped


@wp.func
def _pos_key_31(pos_2d: wp.vec2) -> wp.uint64:
    """Quantize a face-plane position to a 31-bit, never-zero identity key.

    0.5 mm resolution; x covers +/-16.4 m (16 bits), y +/-4.1 m (14 bits),
    both measured from the pair's reference body, and bit 0 is always set so a
    real packed value can never equal the empty-slot sentinel 0.  Positions
    beyond the range clamp, which can only merge far-apart duplicates into a
    tie -- and ties are kept, never dropped.
    """
    # Clamp in FLOAT before converting so extreme coordinates cannot overflow
    # the int32 conversion (they would corrupt the identity, not just clamp it).
    qx = wp.int32(wp.clamp(wp.floor(pos_2d[0] * 2000.0), -32768.0, 32767.0)) + 32768
    qy = wp.int32(wp.clamp(wp.floor(pos_2d[1] * 2000.0), -8192.0, 8191.0)) + 8192
    return (wp.uint64(qx) << wp.uint64(15)) | (wp.uint64(qy) << wp.uint64(1)) | wp.uint64(1)


@wp.func
def _pack_score(primary: float, pos_key: wp.uint64) -> wp.uint64:
    """Pack ``(score, position identity)`` for ``atomic_max`` competition.

    Layout (bit 63 kept zero so the value orders identically read as signed or
    unsigned)::

        [62:31] float_flip(primary)              (32 bits)
        [30:0]  quantized face-plane position    (31 bits, never zero)

    Every contact of a group competes for every spatial slot on projection
    alone -- there is no depth gate or depth preference.  Both were tried and
    both starve patches: a gate drops every contact whose gap spread exceeds
    the window, and a high-order "near" preference is won outright by the
    group's deepest contact (trivially near), which then takes all slots.
    Either way a tilted box face collapses to single-point support and
    diverges.  Pure projection competition gives each direction slot to its
    sampled-support representative (the directional sampling itself is the
    remaining approximation; see the module docstring).

    The low bits are the GEOMETRIC tie-break: the contact's own quantized
    position on the pair-anchored face plane.  Symmetric collider layouts tie
    on the primary score constantly -- a row of foot cylinders shares a
    coordinate exactly -- and the previous tie-break, a hash of contact
    content, flipped winners whenever any input float moved (a static shape's
    witness point is world-space, so it changes under pure translation),
    churning the kept set of a body sliding in a straight line.  The
    pair-anchored key preserves geometric ordering under rigid translation and,
    unlike a single scalar, identifies the contact up to 0.5 mm co-location.
    Quantization can merge identities into a tie under sub-quantum motion; ties
    fail open by keeping every coincident packed duplicate.

    No buffer index is stored: winners identify THEMSELVES in the selection
    pass by comparing their own packed value against the slot, so the winning
    SET is a pure function of contact geometry: invariant to thread scheduling,
    buffer order, and buffer capacity, while rigid translation preserves the
    key ordering.  Two qualifications: contacts that are coincident at the
    0.5 mm quantum all tie and are all kept, so duplicated colliders can exceed
    the per-group slot bound; and near the hashtable's probe budget the OUTCOME
    (reduced vs whole-frame fail-open) can depend on concurrent insertion order
    even though each outcome's kept set is itself deterministic.  Size the
    table so ``fallback_frames`` stays zero.
    """
    return (wp.uint64(float_flip(primary)) << wp.uint64(31)) | pos_key


@wp.func
def _slot_index(entry_idx: int, slot: int) -> int:
    """Address of one value slot, entry-major.

    An entry's slots are adjacent, so each pass that touches all slots of one
    entry -- clearing, spatial competition, winner selection -- works on a
    single contiguous 104-byte run.  Slot-major addressing
    (``slot * capacity + entry``) instead put an entry's slots one whole
    capacity apart, costing one cache line per slot: 33 MB of stride between
    consecutive accesses at G1's 4.2M-entry table.
    """
    return entry_idx * wp.static(BODY_PAIR_REDUCTION_SLOTS) + slot


# Launch width for the passes that walk the contact buffer. Every one of them
# grid-strides over the LIVE contact count, so the launch only has to be wide
# enough to fill the device; sizing it to rigid_contact_max instead spends most
# of its threads on an early-exit test, because a training scene provisions the
# contact buffer far above the count it actually reaches (measured on G1: 102080
# capacity against a peak of 7937 live contacts at 64 envs, a 12.9x margin that
# holds as both scale with env count).
REDUCTION_MAX_THREADS = 262144


@wp.kernel(enable_backward=False)
def _detect_identity_kernel(
    ht_active_slots: wp.array[wp.int32],
    ht_capacity: int,
    # in/out
    frame_state: wp.array[wp.int32],
    stats: wp.array[wp.int32],
):
    """Flag frames where reduction provably cannot remove anything.

    When the number of distinct groups equals the number of live contacts,
    every group holds exactly one contact: the deepest IS the only extreme and
    every contact is kept.  Selection, compaction, and write-back are then
    pure bandwidth waste -- the buffer already contains the result.  This is
    the structural no-benefit regime (single-collider bodies), where the pass
    otherwise measures ~5% end-to-end overhead; the ``identity_frames``
    telemetry doubles as the caller's signal that this scene does not want the
    feature at all.
    """
    work = frame_state[wp.static(FRAME_WORK_COUNT)]
    if work <= 0:
        return
    if frame_state[wp.static(FRAME_FALLBACK)] != 0:
        return
    if ht_active_slots[ht_capacity] == work:
        frame_state[wp.static(FRAME_IDENTITY)] = 1
        wp.atomic_max(stats, wp.static(STAT_CONTACTS_KEPT), work)
        stats[wp.static(STAT_IDENTITY_FRAMES)] = 1


@wp.kernel(enable_backward=False)
def _detect_noop_kernel(
    keep_scan: wp.array[wp.int32],
    # in/out
    frame_state: wp.array[wp.int32],
    stats: wp.array[wp.int32],
):
    """Flag frames where selection kept everything (tier-2 no-op detection).

    The group-count test above catches single-contact groups, but the common
    no-benefit regime -- single-collider bodies like plain boxes -- produces
    groups of up to ~5 contacts that all fit in the slots, so nothing is
    discarded even though groups are larger than one. That is only known after
    the keep scan: when the kept count equals the work count, the compaction
    would copy every contact onto itself, so the copy-back is skipped and the
    buffer is delivered as-is.
    """
    if frame_state[wp.static(FRAME_IDENTITY)] != 0:
        return  # tier 1 already fired
    work = frame_state[wp.static(FRAME_WORK_COUNT)]
    if work <= 0 or frame_state[wp.static(FRAME_INPUT_OVERFLOW)] != 0:
        return
    if frame_state[wp.static(FRAME_FALLBACK)] != 0:
        return  # fallback frames keep everything too, but their accounting
        # (fallback_frames counter) lives in the count-write kernel
    if keep_scan[work - 1] == work:
        frame_state[wp.static(FRAME_IDENTITY)] = 1
        wp.atomic_max(stats, wp.static(STAT_CONTACTS_KEPT), work)
        stats[wp.static(STAT_IDENTITY_FRAMES)] = 1


@wp.func
def _register_contact_one(
    i: int,
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_group: wp.array[wp.int32],
    up_rotation: wp.mat33,
    cell_size: float,
    ht_keys: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    ht_values: wp.array[wp.uint64],
    frame_state: wp.array[wp.int32],
    keep_flags: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    stats: wp.array[wp.int32],
):
    """Enter contact ``i`` into every slot of its group: depth and all extremes.

    Depth and the spatial extremes go in together because they are independent
    ``atomic_max`` accumulations into disjoint slots of the same entry.

    The per-item body lives in a ``wp.func`` so the grid-stride kernel around
    it can bail out of one item without leaving the loop: Warp does not reliably
    honor ``return`` from inside a kernel for-loop.
    """
    # Initialize the flag here rather than with a capacity-wide memset: this
    # pass already visits exactly the live range, and rigid_contact_max is an
    # order of magnitude larger than that.
    keep_flags[i] = 0
    if contact_shape0[i] < 0 or contact_shape1[i] < 0:
        keep_flags[i] = 1
        contact_entry[i] = -1
        return

    key, gap, pos_2d, cell_clamped = _contact_group_key(
        i,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_margin0,
        contact_margin1,
        body_q,
        shape_body,
        shape_group,
        up_rotation,
        cell_size,
    )
    if cell_clamped:
        wp.atomic_add(stats, wp.static(STAT_CELL_CLAMPS), 1)
    entry_idx = _bp_find_or_insert(key, ht_keys, ht_active_slots)
    contact_entry[i] = entry_idx
    if entry_idx < 0:
        # Table budget exhausted. Keeping just THIS contact would make the
        # kept set depend on which threads claimed the last entries, so the
        # whole frame falls back to the unreduced set (see the select pass).
        keep_flags[i] = 1
        frame_state[wp.static(FRAME_FALLBACK)] = 1
        wp.atomic_add(stats, wp.static(STAT_FAIL_OPEN), 1)
        return
    contact_gap[i] = gap
    contact_pos2d[i] = pos_2d

    pos_key = _pos_key_31(pos_2d)
    depth_value = _pack_score(-gap, pos_key)
    slot_idx = _slot_index(entry_idx, wp.static(DEEPEST_SLOT))
    if ht_values[slot_idx] < depth_value:
        wp.atomic_max(ht_values, slot_idx, depth_value)

    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        value = _pack_score(wp.dot(pos_2d, _direction_2d(dir_i)), pos_key)
        dir_slot = _slot_index(entry_idx, dir_i)
        if ht_values[dir_slot] < value:
            wp.atomic_max(ht_values, dir_slot, value)
    if gap <= 0.0:
        for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
            value = _pack_score(wp.dot(pos_2d, _direction_2d(dir_i)), pos_key)
            touch_slot = _slot_index(entry_idx, wp.static(TOUCHING_SLOT_OFFSET) + dir_i)
            if ht_values[touch_slot] < value:
                wp.atomic_max(ht_values, touch_slot, value)


@wp.kernel(enable_backward=False)
def _register_contacts_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_group: wp.array[wp.int32],
    up_rotation: wp.mat33,
    cell_size: float,
    ht_keys: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    ht_values: wp.array[wp.uint64],
    frame_state: wp.array[wp.int32],
    num_threads: int,
    # outputs
    keep_flags: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    stats: wp.array[wp.int32],
):
    """Pass 1 of 2: enter every contact into all slots of its group.

    Caches each contact's hashtable entry, gap and face-plane position so the
    selection pass neither recomputes geometry nor re-probes the table.  A
    contact that cannot be registered (hashtable full or detached shape) is kept
    unconditionally -- reduction must fail open, never drop silently.
    """
    count = frame_state[wp.static(FRAME_WORK_COUNT)]
    i = wp.tid()
    while i < count:
        _register_contact_one(
            i,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_margin0,
            contact_margin1,
            body_q,
            shape_body,
            shape_group,
            up_rotation,
            cell_size,
            ht_keys,
            ht_active_slots,
            ht_values,
            frame_state,
            keep_flags,
            contact_entry,
            contact_gap,
            contact_pos2d,
            stats,
        )
        i += num_threads


@wp.kernel(enable_backward=False)
def _clear_active_values_kernel(
    ht_active_slots: wp.array[wp.int32],
    ht_capacity: int,
    num_threads: int,
    # outputs
    ht_values: wp.array[wp.uint64],
):
    """Zero the value slots of every active hashtable entry from the last step.

    Grid-strides over the ACTIVE entry list, not the table capacity: the table
    is deliberately provisioned well above the group count a scene reaches
    (G1 runs at load 0.08), so a capacity-wide launch spent over 90% of its
    threads deciding they had nothing to clear.
    """
    count = ht_active_slots[ht_capacity]
    t = wp.tid()
    while t < count:
        entry_idx = ht_active_slots[t]
        for slot in range(wp.static(BODY_PAIR_REDUCTION_SLOTS)):
            ht_values[_slot_index(entry_idx, slot)] = wp.uint64(0)
        t += num_threads


@wp.func
def _select_winner_one(
    i: int,
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    ht_values: wp.array[wp.uint64],
    keep_flags: wp.array[wp.int32],
):
    """Flag contact ``i`` if one of the values it submitted won its slot."""
    entry_idx = contact_entry[i]
    if entry_idx < 0:
        return  # keep_flags[i] already set by the register pass (fail open)

    gap = contact_gap[i]
    pos_2d = contact_pos2d[i]

    # NOTE: no early ``return`` inside the loop -- Warp does not reliably honor
    # returns from within a for-loop, so accumulate and flag once.
    pos_key = _pos_key_31(pos_2d)
    won = ht_values[_slot_index(entry_idx, wp.static(DEEPEST_SLOT))] == _pack_score(-gap, pos_key)
    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        value = _pack_score(wp.dot(pos_2d, _direction_2d(dir_i)), pos_key)
        if ht_values[_slot_index(entry_idx, dir_i)] == value:
            won = True
        if gap <= 0.0 and ht_values[_slot_index(entry_idx, wp.static(TOUCHING_SLOT_OFFSET) + dir_i)] == value:
            won = True
    if won:
        keep_flags[i] = 1


@wp.kernel(enable_backward=False)
def _select_winners_kernel(
    contact_count: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    ht_values: wp.array[wp.uint64],
    frame_state: wp.array[wp.int32],
    num_threads: int,
    # outputs
    keep_flags: wp.array[wp.int32],
):
    """Pass 2 of 2: every contact checks whether its own packed value won a slot.

    Winner self-identification: the packed values carry no buffer index, so a
    contact is kept iff one of the values it submitted equals the slot's final
    winner. Two contacts with identical content (equal score AND position
    identity) both match and are both kept -- reduction fails open on true
    ties. The kept set is therefore a pure function of contact geometry:
    invariant to thread scheduling, buffer order, and buffer capacity.
    """
    count = frame_state[wp.static(FRAME_WORK_COUNT)]
    if frame_state[wp.static(FRAME_IDENTITY)] != 0:
        count = 0  # identity frames skip compaction entirely; flags are unused
    keep_all = frame_state[wp.static(FRAME_FALLBACK)] != 0
    i = wp.tid()
    while i < count:
        if keep_all:
            # Table budget was exceeded somewhere this frame: the only
            # deterministic result is the whole unreduced set.
            keep_flags[i] = 1
        else:
            _select_winner_one(i, contact_entry, contact_gap, contact_pos2d, ht_values, keep_flags)
        i += num_threads


@wp.func
def _verify_invariant_one(
    i: int,
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    ht_values: wp.array[wp.uint64],
    keep_flags: wp.array[wp.int32],
    violations: wp.array[wp.int32],
):
    """Re-derive contact ``i``'s keep/discard decision and count disagreements."""
    entry_idx = contact_entry[i]
    if entry_idx < 0:
        return  # fail-open contacts are kept by definition

    gap = contact_gap[i]
    pos_2d = contact_pos2d[i]
    kept = keep_flags[i] != 0

    matched = False
    beaten = False

    pos_key = _pos_key_31(pos_2d)
    deepest_value = ht_values[_slot_index(entry_idx, wp.static(DEEPEST_SLOT))]
    my_depth = _pack_score(-gap, pos_key)
    if my_depth == deepest_value:
        matched = True
    elif my_depth > deepest_value:
        beaten = True  # I out-rank the recorded winner: the slot missed me

    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        value = _pack_score(wp.dot(pos_2d, _direction_2d(dir_i)), pos_key)
        slot_value = ht_values[_slot_index(entry_idx, dir_i)]
        if value == slot_value:
            matched = True
        elif value > slot_value:
            beaten = True
        if gap <= 0.0:
            touch_value = ht_values[_slot_index(entry_idx, wp.static(TOUCHING_SLOT_OFFSET) + dir_i)]
            if value == touch_value:
                matched = True
            elif value > touch_value:
                beaten = True

    if kept and not matched:
        wp.atomic_add(violations, wp.static(STAT_VIOLATIONS), 1)  # kept without winning: too permissive
    if (not kept) and matched:
        wp.atomic_add(violations, wp.static(STAT_VIOLATIONS), 1)  # winner discarded: selection missed it
    if (not kept) and (not matched) and beaten:
        wp.atomic_add(violations, wp.static(STAT_OUTRANKED), 1)  # out-ranks a winner: atomic lost an update


@wp.kernel(enable_backward=False)
def _verify_invariant_kernel(
    contact_count: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    ht_values: wp.array[wp.uint64],
    keep_flags: wp.array[wp.int32],
    frame_state: wp.array[wp.int32],
    num_threads: int,
    # outputs
    violations: wp.array[wp.int32],
):
    """Certificate mode: re-derive the keep/discard decision and count disagreements.

    For every contact, recompute its packed slot values and check the
    invariant the pass promises: a DISCARDED contact must not beat any final
    slot winner it was eligible for, and a KEPT registered contact must
    actually match at least one winner. Any disagreement means a slot race,
    clearing bug, or ranking regression -- counted, never silent.
    """
    count = frame_state[wp.static(FRAME_WORK_COUNT)]
    if frame_state[wp.static(FRAME_FALLBACK)] != 0 or frame_state[wp.static(FRAME_IDENTITY)] != 0:
        count = 0  # keep-all/identity frames are trivially consistent
    i = wp.tid()
    while i < count:
        _verify_invariant_one(i, contact_entry, contact_gap, contact_pos2d, ht_values, keep_flags, violations)
        i += num_threads


@wp.kernel(enable_backward=False)
def _gather_kept_contacts_kernel(
    contact_count: wp.array[wp.int32],
    keep_flags: wp.array[wp.int32],
    keep_scan: wp.array[wp.int32],
    src_point_id: wp.array[wp.int32],
    src_shape0: wp.array[wp.int32],
    src_shape1: wp.array[wp.int32],
    src_point0: wp.array[wp.vec3],
    src_point1: wp.array[wp.vec3],
    src_offset0: wp.array[wp.vec3],
    src_offset1: wp.array[wp.vec3],
    src_normal: wp.array[wp.vec3],
    src_margin0: wp.array[wp.float32],
    src_margin1: wp.array[wp.float32],
    src_tids: wp.array[wp.int32],
    frame_state: wp.array[wp.int32],
    num_threads: int,
    # outputs
    dst_point_id: wp.array[wp.int32],
    dst_shape0: wp.array[wp.int32],
    dst_shape1: wp.array[wp.int32],
    dst_point0: wp.array[wp.vec3],
    dst_point1: wp.array[wp.vec3],
    dst_offset0: wp.array[wp.vec3],
    dst_offset1: wp.array[wp.vec3],
    dst_normal: wp.array[wp.vec3],
    dst_margin0: wp.array[wp.float32],
    dst_margin1: wp.array[wp.float32],
    dst_tids: wp.array[wp.int32],
):
    """Stable-compact kept contacts into the scratch arrays."""
    count = frame_state[wp.static(FRAME_WORK_COUNT)]
    if frame_state[wp.static(FRAME_IDENTITY)] != 0 or frame_state[wp.static(FRAME_FALLBACK)] != 0:
        count = 0  # the buffer already contains the result (no-op or keep-all)
    i = wp.tid()
    while i < count:
        if keep_flags[i] != 0:
            dst = keep_scan[i] - 1  # inclusive scan -> 0-based position
            _copy_contact(
                i,
                dst,
                src_point_id,
                src_shape0,
                src_shape1,
                src_point0,
                src_point1,
                src_offset0,
                src_offset1,
                src_normal,
                src_margin0,
                src_margin1,
                src_tids,
                dst_point_id,
                dst_shape0,
                dst_shape1,
                dst_point0,
                dst_point1,
                dst_offset0,
                dst_offset1,
                dst_normal,
                dst_margin0,
                dst_margin1,
                dst_tids,
            )
        i += num_threads


@wp.func
def _copy_contact(
    i: int,
    dst: int,
    src_point_id: wp.array[wp.int32],
    src_shape0: wp.array[wp.int32],
    src_shape1: wp.array[wp.int32],
    src_point0: wp.array[wp.vec3],
    src_point1: wp.array[wp.vec3],
    src_offset0: wp.array[wp.vec3],
    src_offset1: wp.array[wp.vec3],
    src_normal: wp.array[wp.vec3],
    src_margin0: wp.array[wp.float32],
    src_margin1: wp.array[wp.float32],
    src_tids: wp.array[wp.int32],
    dst_point_id: wp.array[wp.int32],
    dst_shape0: wp.array[wp.int32],
    dst_shape1: wp.array[wp.int32],
    dst_point0: wp.array[wp.vec3],
    dst_point1: wp.array[wp.vec3],
    dst_offset0: wp.array[wp.vec3],
    dst_offset1: wp.array[wp.vec3],
    dst_normal: wp.array[wp.vec3],
    dst_margin0: wp.array[wp.float32],
    dst_margin1: wp.array[wp.float32],
    dst_tids: wp.array[wp.int32],
):
    """Copy one contact record from index ``i`` to index ``dst``."""
    dst_point_id[dst] = src_point_id[i]
    dst_shape0[dst] = src_shape0[i]
    dst_shape1[dst] = src_shape1[i]
    dst_point0[dst] = src_point0[i]
    dst_point1[dst] = src_point1[i]
    dst_offset0[dst] = src_offset0[i]
    dst_offset1[dst] = src_offset1[i]
    dst_normal[dst] = src_normal[i]
    dst_margin0[dst] = src_margin0[i]
    dst_margin1[dst] = src_margin1[i]
    dst_tids[dst] = src_tids[i]


@wp.kernel(enable_backward=False)
def _scatter_back_kernel(
    contact_count: wp.array[wp.int32],
    src_point_id: wp.array[wp.int32],
    src_shape0: wp.array[wp.int32],
    src_shape1: wp.array[wp.int32],
    src_point0: wp.array[wp.vec3],
    src_point1: wp.array[wp.vec3],
    src_offset0: wp.array[wp.vec3],
    src_offset1: wp.array[wp.vec3],
    src_normal: wp.array[wp.vec3],
    src_margin0: wp.array[wp.float32],
    src_margin1: wp.array[wp.float32],
    src_tids: wp.array[wp.int32],
    frame_state: wp.array[wp.int32],
    num_threads: int,
    # outputs
    dst_point_id: wp.array[wp.int32],
    dst_shape0: wp.array[wp.int32],
    dst_shape1: wp.array[wp.int32],
    dst_point0: wp.array[wp.vec3],
    dst_point1: wp.array[wp.vec3],
    dst_offset0: wp.array[wp.vec3],
    dst_offset1: wp.array[wp.vec3],
    dst_normal: wp.array[wp.vec3],
    dst_margin0: wp.array[wp.float32],
    dst_margin1: wp.array[wp.float32],
    dst_tids: wp.array[wp.int32],
):
    """Write the compacted contacts back into the live arrays.

    Runs after ``_write_reduced_count_kernel``, so ``contact_count`` already
    holds the KEPT count: only that range is touched, instead of copying
    entire capacity-sized arrays back.  Overflow, identity, and fallback
    frames all zero the bound through the frame state: on overflow the counter
    deliberately holds its raw over-capacity value, and on identity/fallback
    the buffer already contains the result, so nothing is written back.
    """
    count = contact_count[0]
    if (
        frame_state[wp.static(FRAME_INPUT_OVERFLOW)] != 0
        or frame_state[wp.static(FRAME_IDENTITY)] != 0
        or frame_state[wp.static(FRAME_FALLBACK)] != 0
    ):
        count = 0
    i = wp.tid()
    while i < count:
        _copy_contact(
            i,
            i,
            src_point_id,
            src_shape0,
            src_shape1,
            src_point0,
            src_point1,
            src_offset0,
            src_offset1,
            src_normal,
            src_margin0,
            src_margin1,
            src_tids,
            dst_point_id,
            dst_shape0,
            dst_shape1,
            dst_point0,
            dst_point1,
            dst_offset0,
            dst_offset1,
            dst_normal,
            dst_margin0,
            dst_margin1,
            dst_tids,
        )
        i += num_threads


@wp.kernel(enable_backward=False)
def _write_reduced_count_kernel(
    keep_scan: wp.array[wp.int32],
    ht_active_slots: wp.array[wp.int32],
    ht_capacity: int,
    frame_state: wp.array[wp.int32],
    # in/out
    contact_count: wp.array[wp.int32],
    stats: wp.array[wp.int32],
    stats64: wp.array[wp.int64],
):
    """Replace the contact count with the number of kept contacts.

    Also records the watermarks that size the pass (how many contacts arrived,
    how many survived, how many hashtable entries the scene actually needed),
    the paired whole-run totals that give the achieved reduction ratio, and all
    lifetime event counters.  Event producers use per-frame int32 atomics; this
    single-thread launch folds those values into int64 totals before the next
    frame clears them.  Overflow frames count toward ``total_frames`` but are
    excluded from both contact sums: their materialized prefix contains holes,
    so no coherent in/kept pair exists for them.
    """
    wp.atomic_max(stats, wp.static(STAT_CONTACTS_IN), contact_count[0])
    wp.atomic_max(stats, wp.static(STAT_ENTRY_WATERMARK), ht_active_slots[ht_capacity])
    stats[wp.static(STAT_FALLBACK_FRAMES)] = frame_state[wp.static(FRAME_FALLBACK)]
    stats64[wp.static(STAT64_TOTAL_FRAMES)] = stats64[wp.static(STAT64_TOTAL_FRAMES)] + wp.int64(1)
    stats64[wp.static(STAT64_PROBE_FAILURES)] = stats64[wp.static(STAT64_PROBE_FAILURES)] + wp.int64(
        stats[wp.static(STAT_FAIL_OPEN)]
    )
    stats64[wp.static(STAT64_CELL_CLAMP_EVENTS)] = stats64[wp.static(STAT64_CELL_CLAMP_EVENTS)] + wp.int64(
        stats[wp.static(STAT_CELL_CLAMPS)]
    )
    stats64[wp.static(STAT64_INVARIANT_VIOLATIONS)] = stats64[wp.static(STAT64_INVARIANT_VIOLATIONS)] + wp.int64(
        stats[wp.static(STAT_VIOLATIONS)]
    )
    stats64[wp.static(STAT64_OUTRANKED_DISCARDS)] = stats64[wp.static(STAT64_OUTRANKED_DISCARDS)] + wp.int64(
        stats[wp.static(STAT_OUTRANKED)]
    )
    stats64[wp.static(STAT64_INPUT_OVERFLOW_FRAMES)] = stats64[wp.static(STAT64_INPUT_OVERFLOW_FRAMES)] + wp.int64(
        stats[wp.static(STAT_INPUT_OVERFLOWS)]
    )
    stats64[wp.static(STAT64_FALLBACK_FRAMES)] = stats64[wp.static(STAT64_FALLBACK_FRAMES)] + wp.int64(
        stats[wp.static(STAT_FALLBACK_FRAMES)]
    )
    stats64[wp.static(STAT64_IDENTITY_FRAMES)] = stats64[wp.static(STAT64_IDENTITY_FRAMES)] + wp.int64(
        stats[wp.static(STAT_IDENTITY_FRAMES)]
    )
    if frame_state[wp.static(FRAME_INPUT_OVERFLOW)] != 0:
        # Leave the raw counter untouched so the solver receives exactly what
        # an unreduced pipeline would deliver and overflow stays observable.
        return
    work = frame_state[wp.static(FRAME_WORK_COUNT)]
    if frame_state[wp.static(FRAME_IDENTITY)] != 0:
        # The count is already the kept count.
        stats64[wp.static(STAT64_SUM_CONTACTS_IN)] = stats64[wp.static(STAT64_SUM_CONTACTS_IN)] + wp.int64(work)
        stats64[wp.static(STAT64_SUM_CONTACTS_KEPT)] = stats64[wp.static(STAT64_SUM_CONTACTS_KEPT)] + wp.int64(work)
        return
    if frame_state[wp.static(FRAME_FALLBACK)] != 0:
        # Keep-all frames deliver the buffer untouched; the counter already
        # equals the kept count, so only the accounting runs.
        wp.atomic_max(stats, wp.static(STAT_CONTACTS_KEPT), work)
        stats64[wp.static(STAT64_SUM_CONTACTS_IN)] = stats64[wp.static(STAT64_SUM_CONTACTS_IN)] + wp.int64(work)
        stats64[wp.static(STAT64_SUM_CONTACTS_KEPT)] = stats64[wp.static(STAT64_SUM_CONTACTS_KEPT)] + wp.int64(work)
        return
    if work > 0:
        kept = keep_scan[work - 1]
        contact_count[0] = kept
        wp.atomic_max(stats, wp.static(STAT_CONTACTS_KEPT), kept)
        stats64[wp.static(STAT64_SUM_CONTACTS_IN)] = stats64[wp.static(STAT64_SUM_CONTACTS_IN)] + wp.int64(work)
        stats64[wp.static(STAT64_SUM_CONTACTS_KEPT)] = stats64[wp.static(STAT64_SUM_CONTACTS_KEPT)] + wp.int64(kept)


@wp.kernel(enable_backward=False)
def _gather_reduced_sort_keys(
    frame_state: wp.array[wp.int32],
    keep_flags: wp.array[wp.int32],
    keep_scan: wp.array[wp.int32],
    source: wp.array[wp.int64],
    destination: wp.array[wp.int64],
    num_threads: int,
):
    """Compact identity keys with precisely the same selection as the contacts."""
    if frame_state[wp.static(FRAME_IDENTITY)] != 0 or frame_state[wp.static(FRAME_FALLBACK)] != 0:
        return
    for i in range(wp.tid(), frame_state[wp.static(FRAME_WORK_COUNT)], num_threads):
        if keep_flags[i] != 0:
            destination[keep_scan[i] - 1] = source[i]


@wp.kernel(enable_backward=False)
def _scatter_reduced_sort_keys(
    frame_state: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    source: wp.array[wp.int64],
    destination: wp.array[wp.int64],
    num_threads: int,
):
    """Install compacted identity keys without an in-place gather race."""
    if (
        frame_state[wp.static(FRAME_IDENTITY)] != 0
        or frame_state[wp.static(FRAME_FALLBACK)] != 0
        or frame_state[wp.static(FRAME_INPUT_OVERFLOW)] != 0
    ):
        return
    for i in range(wp.tid(), contact_count[0], num_threads):
        destination[i] = source[i]


class BodyPairContactReducer:
    """Owns the buffers and launch sequence for body-pair contact reduction.

    Created by :class:`newton.CollisionPipeline` from
    ``CollisionPipeline.BodyPairReductionConfig``; not part of the public API.

    Args:
        rigid_contact_max: Capacity of the ``Contacts`` buffer being reduced.
        cell_size: Spatial cell edge [m] on the normal bin's face plane. Each
            (body pair, bin, cell) keeps its own deepest + extremes, so
            same-normal patches farther apart than a cell never compete.
        device: Warp device.
    """

    def __init__(
        self,
        rigid_contact_max: int,
        cell_size: float,
        device,
        shape_group,
        up_axis: int = 2,
        hashtable_headroom: float = 1.0,
        group_pair_bound: int | None = None,
        borrowed_scratch: dict | None = None,
        preserve_sort_keys: bool = False,
    ):
        self.rigid_contact_max = rigid_contact_max
        # CollisionPipeline.BodyPairReductionConfig validates this at float32 precision.
        self.cell_size = float(np.float32(cell_size))
        self.device = device
        hashtable_headroom = float(hashtable_headroom)
        # Per-shape reduction-group ids from build_reduction_groups (material-
        # equivalence classes), and the exact rotation mapping the model's up
        # axis into the bin table's Z-up frame.
        self.shape_group = wp.array(shape_group, dtype=wp.int32, device=device)
        self.up_rotation = wp.mat33(*(v for row in _up_axis_rotation(int(up_axis)) for v in row))
        self._sort_key_scratch = (
            wp.zeros(rigid_contact_max, dtype=wp.int64, device=device) if preserve_sort_keys else None
        )
        # Re-derives every keep/discard decision each frame; for tests and debugging.
        self.verify = False
        # Telemetry storage: int32 watermarks/per-frame atomic scratch plus
        # int64 lifetime totals; see stats() for the meanings.
        self._stats = wp.zeros(STAT_COUNT, dtype=wp.int32, device=device)
        self._stats64 = wp.zeros(STAT64_COUNT, dtype=wp.int64, device=device)
        # Per-frame device state: validated work count + overflow/fallback
        # flags, reset by the preparation kernel each reduce.
        self._frame_state = wp.zeros(FRAME_STATE_COUNT, dtype=wp.int32, device=device)
        # One entry per (body pair, bin, cell) actually touched. Sized from the
        # scene's own topology -- the distinct group pairs its contact-pair list
        # can produce (see build_reduction_group_pair_bound) -- rather than from
        # rigid_contact_max, which counts CONTACTS and relates to the group
        # count only through how many contacts share a patch, i.e. through
        # exactly the quantity this pass exists to vary.
        #
        # The anchor is loose in both directions and deliberately so: it bounds
        # only the pair dimension of the key, and reachable pairs are not
        # simultaneously active. Neither direction is unsafe. Undersizing
        # cannot drop a contact -- a bounded probe failure makes the whole
        # frame keep everything, counted in fallback_frames -- and
        # rigid_contact_max caps the request because a group cannot exist
        # without a contact to create it.
        if group_pair_bound is not None and group_pair_bound > 0:
            # A replicated scene can have only one reachable pair per body,
            # yet each pair occupies several normal bins / spatial cells.
            # Reserve a linear budget of four patch entries per material group
            # at half load, in addition to the reachable-pair estimate. This
            # preserves headroom as isolated environments are replicated without
            # multiplying a dense candidate-pair graph by the same factor.
            group_budget = 8 * len(np.unique(shape_group))
            capacity_request = int(max(group_pair_bound, group_budget) * hashtable_headroom)
        else:
            # No precomputed pair list to derive from; fall back to a
            # contact-anchored heuristic rather than guessing.
            capacity_request = int(rigid_contact_max * 0.25 * hashtable_headroom)
        self.hashtable_capacity_request = max(1024, min(capacity_request, rigid_contact_max))
        self.group_pair_bound = int(group_pair_bound) if group_pair_bound is not None else -1
        self.hashtable = HashTable(self.hashtable_capacity_request, device=device)
        self.ht_values = wp.zeros(BODY_PAIR_REDUCTION_SLOTS * self.hashtable.capacity, dtype=wp.uint64, device=device)
        # Fixed launch width for every pass; the kernels grid-stride over the
        # live count, so this only has to fill the device. See
        # REDUCTION_MAX_THREADS.
        self.stride_threads = min(rigid_contact_max, REDUCTION_MAX_THREADS)
        # The clearing kernel walks the ACTIVE entry list, which holds as many
        # entries as the scene has groups -- thousands, not millions -- so its
        # launch is sized an order of magnitude narrower than the contact passes.
        self.entry_stride_threads = min(self.hashtable.capacity, 32768)
        self.keep_flags = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        self.keep_scan = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        # pass-1 cache read by the selection pass: hashtable entry, canonical
        # gap, face-plane position
        self.contact_entry = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        self.contact_gap = wp.zeros(rigid_contact_max, dtype=wp.float32, device=device)
        self.contact_pos2d = wp.zeros(rigid_contact_max, dtype=wp.vec2, device=device)
        # Full-size gather scratch, borrowed from the deterministic sorter when
        # there is one (see ContactSorter.borrow_full_scratch): the two stages
        # run strictly sequentially inside collide().
        dtypes = {
            "shape0": wp.int32,
            "shape1": wp.int32,
            "point0": wp.vec3,
            "point1": wp.vec3,
            "offset0": wp.vec3,
            "offset1": wp.vec3,
            "normal": wp.vec3,
            "margin0": wp.float32,
            "margin1": wp.float32,
            "tids": wp.int32,
        }
        borrowed = borrowed_scratch or {}
        self._borrowed_scratch = {
            name: borrowed[name] for name in dtypes if name in borrowed and borrowed[name].shape[0] >= rigid_contact_max
        }
        self._scratch = {
            name: (
                self._borrowed_scratch[name]
                if name in self._borrowed_scratch
                else wp.zeros(rigid_contact_max, dtype=dtype, device=device)
            )
            for name, dtype in dtypes.items()
        }
        self._scratch["point_id"] = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)

    def reduce(self, model, state, contacts, *, sort_keys=None):
        """Compact ``contacts`` in place, dropping patch-redundant candidates.

        Args:
            model: The simulation model.
            state: Current state (body transforms for witness-point math).
            contacts: The contacts buffer to compact.
            sort_keys: Optional keys aligned with the input contact records.
                Compact them with the same stable selection; requires
                ``preserve_sort_keys=True`` at construction.
        """
        sc = self._scratch

        # Derive this frame's validated work count (see the kernel docstring:
        # the raw narrow-phase counter is NOT a safe array bound on overflow).
        wp.launch(
            _prepare_frame_kernel,
            dim=1,
            inputs=[contacts.rigid_contact_count, self.rigid_contact_max],
            outputs=[self._frame_state, self._stats],
            device=self.device,
            record_tape=False,
        )

        # keep_flags needs no memset: pass 1 writes every live entry, and
        # positions past the live count are never read (the scan output above
        # the count is unused, and the gather only walks the live range).
        #
        # Clear only the previously-active entries' value slots, then their keys
        # (order matters: the value clear reads the active list that the key
        # clear resets). Zeroing the full slot array instead costs a ~200 MB
        # memset per collide at large env counts.
        wp.launch(
            _clear_active_values_kernel,
            dim=self.entry_stride_threads,
            inputs=[self.hashtable.active_slots, self.hashtable.capacity, self.entry_stride_threads],
            outputs=[self.ht_values],
            device=self.device,
            record_tape=False,
        )
        self.hashtable.clear_active(record_tape=False)

        wp.launch(
            _register_contacts_kernel,
            dim=self.stride_threads,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                state.body_q,
                model.shape_body,
                self.shape_group,
                self.up_rotation,
                self.cell_size,
                self.hashtable.keys,
                self.hashtable.active_slots,
                self.ht_values,
                self._frame_state,
                self.stride_threads,
            ],
            outputs=[
                self.keep_flags,
                self.contact_entry,
                self.contact_gap,
                self.contact_pos2d,
                self._stats,
            ],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _detect_identity_kernel,
            dim=1,
            inputs=[self.hashtable.active_slots, self.hashtable.capacity],
            outputs=[self._frame_state, self._stats],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _select_winners_kernel,
            dim=self.stride_threads,
            inputs=[
                contacts.rigid_contact_count,
                self.contact_entry,
                self.contact_gap,
                self.contact_pos2d,
                self.ht_values,
                self._frame_state,
                self.stride_threads,
            ],
            outputs=[self.keep_flags],
            device=self.device,
            record_tape=False,
        )
        if self.verify:
            wp.launch(
                _verify_invariant_kernel,
                dim=self.stride_threads,
                inputs=[
                    contacts.rigid_contact_count,
                    self.contact_entry,
                    self.contact_gap,
                    self.contact_pos2d,
                    self.ht_values,
                    self.keep_flags,
                    self._frame_state,
                    self.stride_threads,
                ],
                outputs=[self._stats],
                device=self.device,
                record_tape=False,
            )
        wp.utils.array_scan(self.keep_flags, self.keep_scan, True)
        wp.launch(
            _detect_noop_kernel,
            dim=1,
            inputs=[self.keep_scan],
            outputs=[self._frame_state, self._stats],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _gather_kept_contacts_kernel,
            dim=self.stride_threads,
            inputs=[
                contacts.rigid_contact_count,
                self.keep_flags,
                self.keep_scan,
                contacts.rigid_contact_point_id,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_tids,
                self._frame_state,
                self.stride_threads,
            ],
            outputs=[
                sc["point_id"],
                sc["shape0"],
                sc["shape1"],
                sc["point0"],
                sc["point1"],
                sc["offset0"],
                sc["offset1"],
                sc["normal"],
                sc["margin0"],
                sc["margin1"],
                sc["tids"],
            ],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _write_reduced_count_kernel,
            dim=1,
            inputs=[self.keep_scan, self.hashtable.active_slots, self.hashtable.capacity, self._frame_state],
            outputs=[contacts.rigid_contact_count, self._stats, self._stats64],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _scatter_back_kernel,
            dim=self.stride_threads,
            inputs=[
                contacts.rigid_contact_count,
                sc["point_id"],
                sc["shape0"],
                sc["shape1"],
                sc["point0"],
                sc["point1"],
                sc["offset0"],
                sc["offset1"],
                sc["normal"],
                sc["margin0"],
                sc["margin1"],
                sc["tids"],
                self._frame_state,
                self.stride_threads,
            ],
            outputs=[
                contacts.rigid_contact_point_id,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_tids,
            ],
            device=self.device,
            record_tape=False,
        )

        if sort_keys is not None:
            if self._sort_key_scratch is None:
                raise ValueError("sort-key compaction requires preserve_sort_keys at construction")
            wp.launch(
                _gather_reduced_sort_keys,
                dim=self.stride_threads,
                inputs=[
                    self._frame_state,
                    self.keep_flags,
                    self.keep_scan,
                    sort_keys,
                    self._sort_key_scratch,
                    self.stride_threads,
                ],
                device=self.device,
                record_tape=False,
            )
            wp.launch(
                _scatter_reduced_sort_keys,
                dim=self.stride_threads,
                inputs=[
                    self._frame_state,
                    contacts.rigid_contact_count,
                    self._sort_key_scratch,
                    sort_keys,
                    self.stride_threads,
                ],
                device=self.device,
                record_tape=False,
            )

    def stats(self) -> dict:
        """Whole-run reduction telemetry (forces a device sync; read outside capture).

        Every public value is a whole-run accumulator -- a max watermark or an
        int64 total -- until :meth:`clear_stats` is called, so one read at the
        end of a run characterizes it. The three sizing watermarks exist so a
        scene that has outgrown its provisioning says so instead of quietly
        reducing less well.

        Returns:
            ``invariant_violations``: disagreements found by the opt-in
            :attr:`verify` re-derivation (0 when verify is off).
            ``probe_failures``: group-table insertions that could not find or
            create their key within the bounded probe budget. Each one flags
            its WHOLE frame for the deterministic keep-all fallback, so this
            counts trigger events, not kept contacts. A failure can mean a full
            table or a long hash cluster; sustained non-zero values usually
            call for a larger ``hashtable_headroom``, but inspect occupancy too.
            ``outranked_discards``: discarded contacts that out-rank their
            slot's recorded winner (verify mode only); non-zero means an atomic
            lost an update.
            ``cell_clamp_events``: contacts whose spatial cell hit the packed
            +/-31 range; non-zero means distant regions of one shape pair are
            merging on the periphery.
            ``max_contacts_in`` / ``max_contacts_kept``: peak live contact count
            before and after reduction.  ``max_contacts_in`` -- NOT kept -- is
            the minimum observed safe ``rigid_contact_max``: the narrow phase
            writes the unreduced candidates into the same buffer before the
            reduction runs.  The two watermarks are accumulated independently
            and may come from different frames, so their ratio is indicative,
            not an exact per-frame reduction ratio -- use the sums below for
            the achieved ratio.
            ``total_frames``: reduce() calls since construction or the last
            :meth:`clear_stats` (int64).
            ``sum_contacts_in`` / ``sum_contacts_kept``: paired whole-run
            totals (int64); ``sum_contacts_kept / sum_contacts_in`` is the
            achieved reduction ratio.  Input-overflow frames are counted in
            ``total_frames`` but excluded from both sums (their materialized
            prefix contains holes, so no coherent in/kept pair exists).
            ``max_hashtable_entries`` / ``hashtable_capacity``: peak distinct
            (body pair, bin, cell) groups against the table that holds them.
            ``hashtable_load``: the ratio of those two -- linear probing
            degrades well past ~0.7. Probe exhaustion may occur below 1.0 when
            one collision cluster exceeds the bounded probe budget.
            ``input_overflow_frames``: frames whose narrow-phase output
            exceeded the buffer capacity; the reduction skipped those frames
            entirely and delivered the raw result (raise
            ``rigid_contact_max``).  Overflow frames are excluded from the
            kept watermark: their materialized prefix contains holes, so no
            meaningful kept count exists for them.
            ``fallback_frames``: frames that deterministically kept the whole
            unreduced set because a group key exhausted the bounded probe
            budget. This can mean a full table or a long collision cluster;
            inspect load and usually raise the hashtable factor.
            ``identity_frames``: frames where reduction provably removed
            nothing; a large fraction means this scene does not benefit from
            the feature and it should be disabled.
        """
        self._raise_if_capturing("stats()")
        v = self._stats.numpy()
        v64 = self._stats64.numpy()
        entries = int(v[STAT_ENTRY_WATERMARK])
        probe_failures = int(v64[STAT64_PROBE_FAILURES])
        return {
            "invariant_violations": int(v64[STAT64_INVARIANT_VIOLATIONS]),
            "probe_failures": probe_failures,
            "outranked_discards": int(v64[STAT64_OUTRANKED_DISCARDS]),
            "cell_clamp_events": int(v64[STAT64_CELL_CLAMP_EVENTS]),
            "max_contacts_in": int(v[STAT_CONTACTS_IN]),
            "max_contacts_kept": int(v[STAT_CONTACTS_KEPT]),
            "max_hashtable_entries": entries,
            "hashtable_capacity": self.hashtable.capacity,
            "hashtable_load": entries / self.hashtable.capacity,
            "input_overflow_frames": int(v64[STAT64_INPUT_OVERFLOW_FRAMES]),
            "fallback_frames": int(v64[STAT64_FALLBACK_FRAMES]),
            "identity_frames": int(v64[STAT64_IDENTITY_FRAMES]),
            "total_frames": int(v64[STAT64_TOTAL_FRAMES]),
            "sum_contacts_in": int(v64[STAT64_SUM_CONTACTS_IN]),
            "sum_contacts_kept": int(v64[STAT64_SUM_CONTACTS_KEPT]),
        }

    def _raise_if_capturing(self, what: str):
        """Reject host-side telemetry operations during CUDA graph capture.

        ``stats()`` would fail on its device sync anyway, but cryptically;
        ``clear_stats()`` would silently RECORD its zeroing into the graph and
        wipe telemetry on every replay.
        """
        if self.device.is_cuda and self.device.is_capturing:
            raise RuntimeError(f"{what} is host-side telemetry; call it outside CUDA graph capture")

    def clear_stats(self):
        """Zero the whole-run telemetry accumulators (host-side, outside capture).

        Lets long training runs isolate per-phase telemetry. Watermarks are
        int32 because the underlying capacities/counts are int32; every
        additive public counter is int64.
        """
        self._raise_if_capturing("clear_stats()")
        self._stats.zero_()
        self._stats64.zero_()

    def describe(self) -> dict:
        """Currently allocated footprint of the pass: buffer bytes by role.

        Fixed at construction.

        Reported per role rather than as one total so an over-provisioned
        ``rigid_contact_max`` is distinguishable from the pass's own overhead;
        ``gather_scratch_owned_bytes`` counts only arrays this pass allocated
        itself, excluding any borrowed from the deterministic sorter.
        """
        n = self.rigid_contact_max
        owned = 0
        borrowed_ids = {id(a) for a in self._borrowed_scratch.values()}
        for arr in self._scratch.values():
            if id(arr) not in borrowed_ids:
                owned += arr.size * wp.types.type_size_in_bytes(arr.dtype)
        borrowed = sum(arr.size * wp.types.type_size_in_bytes(arr.dtype) for arr in self._borrowed_scratch.values())
        return {
            "rigid_contact_max": n,
            "hashtable_capacity": self.hashtable.capacity,
            # The topology anchor the capacity was derived from (-1 when the
            # model carried no contact-pair list and the contact-anchored
            # fallback was used). Compare against stats()["max_hashtable_entries"]
            # to see how much of the anchor's slack this scene actually needs.
            "group_pair_bound": self.group_pair_bound,
            "hashtable_capacity_request": self.hashtable_capacity_request,
            "slots_per_entry": BODY_PAIR_REDUCTION_SLOTS,
            "hashtable_bytes": self.hashtable.keys.size * 8 + self.hashtable.active_slots.size * 4,
            "slot_value_bytes": self.ht_values.size * 8,
            "per_contact_cache_bytes": n * (4 + 4 + 4 + 4 + 8),
            "gather_scratch_owned_bytes": owned,
            "gather_scratch_borrowed_bytes": borrowed,
            "sort_key_scratch_bytes": 0 if self._sort_key_scratch is None else self._sort_key_scratch.size * 8,
        }
