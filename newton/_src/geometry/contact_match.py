# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame-to-frame contact matching via binary search on sorted contact keys.

Given the previous frame's sorted contacts (keys, world-space midpoints,
normals) and the current frame's unsorted contacts, this module finds
correspondences using the deterministic sort key from
:func:`~newton._src.geometry.contact_data.make_contact_sort_key`.

For each new contact the matcher binary-searches the previous frame's
sorted keys for the ``(shape_a, shape_b)`` pair range — ignoring the
``sort_sub_key`` bits — then picks the closest previous contact in that
range whose normal also passes the dot-product threshold.  "Closest" is
measured in world space between the contact *midpoints*
``0.5 * (world(point0) + world(point1))``, i.e. symmetric in shape 0 and
shape 1.  The result is a per-contact match index:

- ``>= 0``: index of the matched contact in the previous frame's sorted buffer.
- ``MATCH_NOT_FOUND (-1)``: shape pair has no prior contacts.
- ``MATCH_BROKEN (-2)``: shape pair exists but no contact within
  position/normal thresholds, *or* a closer new contact won the same
  prev contact in the uniqueness resolve pass.

Why ignore sort_sub_key
-----------------------
Multi-contact manifolds (e.g. box-box face-face) can rotate the
``sort_sub_key`` assignment across frames when their internal generation
order shifts (e.g. the Sutherland-Hodgman clip's starting vertex moves
by one slot), even though the physical contact points stay essentially
in place.  Matching on the full key would mark these contacts broken
every frame.  Pair counts are small (a few manifold points per pair),
so the linear scan inside the pair range is cheap.

One-to-one match via packed atomic_min
--------------------------------------
A pair-range scan can have multiple new contacts pick the same prev
contact as their closest.  To keep the mapping injective without
sorting or CAS retries, the matcher uses a single ``wp.atomic_min`` per
new contact on a per-prev ``int64`` claim word:

    claim = (float_flip(dist_sq) << 32) | (sort_key & 0xFFFFFFFF)

``float_flip`` reinterprets the non-negative ``dist_sq`` as a
sortable ``uint32``, so the high 32 bits order claims by ascending
distance; the low 32 bits hold the low 32 bits of the new contact's
sort key (which uniquely identify it within its shape pair), breaking
ties deterministically.  Using the sort key rather than the
``wp.tid()`` of the new contact keeps the winner invariant under the
non-deterministic unsorted slot assignment that the narrow phase gives
us via ``wp.atomic_add`` -- two parallel runs of the same scene that
emit the same set of new contacts in different orders will pick the
same winner.  See :func:`_pack_claim` for the per-pair uniqueness
caveat (multi-contact paths are unique by construction; the global
reduction path is no worse than the upstream deterministic sort).

After the match kernel runs, a small finalize kernel reads
``prev_claim[best_idx]`` and demotes any new contact whose sort-key low
32 bits do not appear in the low bits of the claim word to
:data:`MATCH_BROKEN`.  Losers are *not* re-matched against a
second-closest prev (kept for simplicity and speed).

Cost: one ``int64[capacity]`` buffer, one ``wp.atomic_min`` per new
contact, and one short finalize kernel launch.  No ``atomic_cas``.

Memory efficiency
-----------------
The matcher reuses the :class:`ContactSorter`'s existing scratch buffers
(:attr:`ContactSorter.scratch_pos_world`, :attr:`ContactSorter.scratch_normal`)
to store previous-frame world-space contact midpoints and normals between
frames.  This works for the *match* kernel because matching runs **before**
``ContactSorter.sort_full``, so the scratch buffers still hold the previous
frame's saved data; ``save_sorted_state`` runs **after** sorting and
refreshes them in-place for the next frame.  The only additional
per-contact allocation for the non-sticky path is the ``_prev_sorted_keys``
buffer (8 bytes/contact) since the sorter's key buffer is overwritten by
``_prepare_sort`` each frame.

Sticky mode needs one extra dedicated buffer (``_prev_normal_sticky``,
12 bytes/contact) because :meth:`replay_matched` runs **after**
``sort_full``, at which point the sorter's ``scratch_normal`` has been
clobbered by the sort's backup pass and no longer contains the previous
frame's sorted normals.  The body-frame point/offset columns already use
dedicated sticky buffers for the same reason.

Per-frame call order (inside :class:`~newton.CollisionPipeline`)::

    matcher.match(...)  # before ContactSorter.sort_full()
    sorter.sort_full(...)  # match_index is permuted with contacts
    matcher.replay_matched(...)  # sticky-only; overwrite matched rows
    matcher.build_report(...)  # optional; must precede save_sorted_state
    matcher.save_sorted_state(...)  # after sorting, replay, and report

The ordering matters: ``save_sorted_state`` overwrites ``_prev_count`` with
the current frame's count, while ``build_report`` reads the *old*
``_prev_count`` to bound the broken-contact enumeration, and sticky
``replay_matched`` must see the post-sort ``match_index`` and the pre-save
``_prev_*`` buffers it reads from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..core.types import Devicelike
from .contact_sort import SORT_KEY_SENTINEL

if TYPE_CHECKING:
    from .contact_sort import ContactSorter

MATCH_NOT_FOUND = wp.constant(wp.int32(-1))
"""Sentinel: no matching key found in last frame's contacts."""

MATCH_BROKEN = wp.constant(wp.int32(-2))
"""Sentinel: key found but position or normal threshold exceeded."""


# ------------------------------------------------------------------
# Warp helpers
# ------------------------------------------------------------------


# Sentinel value for unclaimed slots in ``_prev_claim``.  Larger than
# any packed (flipped_dist << 32 | key_low32) any kernel will ever
# produce, so the first ``atomic_min`` always wins.
_CLAIM_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))


@wp.func
def _lower_bound_int64(
    lower: int,
    upper: int,
    target: wp.int64,
    keys: wp.array[wp.int64],
) -> int:
    """First index in ``keys[lower:upper]`` whose value is >= *target*.

    Returns ``upper`` if no such index exists.
    """
    left = lower
    right = upper
    while left < right:
        mid = left + (right - left) // 2
        if keys[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


@wp.func_native("""
uint32_t i = reinterpret_cast<uint32_t&>(f);
uint32_t mask = (uint32_t)(-(int)(i >> 31)) | 0x80000000u;
return i ^ mask;
""")
def _float_flip(f: float) -> wp.uint32:
    """Reinterpret a 32-bit float as a sortable ``uint32`` (Stereopsis trick).

    For non-negative floats this is a strictly monotone encoding, so
    comparing the resulting ``uint32`` orders the original floats
    correctly.  We only ever feed non-negative ``dist_sq`` values into
    it, so the negative branch is unused here but kept generic.
    """
    ...


@wp.func
def _pack_claim(dist_sq: float, key_low32: wp.int64) -> wp.int64:
    """Pack ``(dist_sq, sort_key_low32)`` into a single int64 for ``atomic_min``.

    High 32 bits: ``float_flip(dist_sq)`` — ascending by distance.
    Low 32 bits:  the low 32 bits of the contact's sort key — deterministic
        tie-break (smallest wins).  Using the sort key (rather than the
        unsorted thread id) keeps the resolution invariant under
        non-deterministic narrow-phase slot assignment: two new contacts
        racing for the same prev contact get the same packed claim
        regardless of which unsorted slot the narrow phase happened to
        hand them this run.

    Within a single shape pair the upper 40 bits of every contact's sort
    key are identical, so the low 32 bits hold the (shape_b LSBs +
    sort_sub_key) which uniquely identifies each contact in the pair as
    long as ``sort_sub_key`` is unique per contact within the pair.

    Note this is a *shared* assumption with the deterministic radix sort
    upstream, not a hard guarantee enforced by it.  The multi-contact
    and mesh/SDF paths build ``sort_sub_key`` from per-contact identifiers
    (clip-vertex slot, triangle/edge/vertex index) that are unique per
    pair by construction, but the reduced-contact path
    (``contact_reduction_global.export_reduced_contacts_kernel``)
    re-uses the original contact's fingerprint as ``sort_sub_key`` and
    only deduplicates by ``contact_id``, so two reduced contacts in the
    same pair can in principle land in different reduction slots and
    still share a fingerprint.  When that happens the deterministic
    sort and this tiebreak degrade together: the contacts are
    indistinguishable to either, and frame-to-frame matching becomes
    order-sensitive only to the same extent the sort itself does.  In
    other words, this scheme is no worse than what the upstream sort
    already provides.
    """
    flipped = wp.int64(_float_flip(dist_sq))
    return (flipped << wp.int64(32)) | (key_low32 & wp.int64(0xFFFFFFFF))


# ------------------------------------------------------------------
# Match kernel
# ------------------------------------------------------------------


@wp.struct
class _MatchData:
    """Bundled arrays for the contact match kernel."""

    # Previous frame (sorted) — pos/normal reuse ContactSorter scratch buffers.
    # ``prev_pos_world`` holds the world-space *midpoint* between shape 0's and
    # shape 1's contact points, saved by the previous frame's save kernel.
    prev_keys: wp.array[wp.int64]
    prev_pos_world: wp.array[wp.vec3]
    prev_normal: wp.array[wp.vec3]
    prev_count: wp.array[wp.int32]

    # Current frame (unsorted).
    new_keys: wp.array[wp.int64]
    new_point0: wp.array[wp.vec3]
    new_point1: wp.array[wp.vec3]
    new_shape0: wp.array[wp.int32]
    new_shape1: wp.array[wp.int32]
    new_normal: wp.array[wp.vec3]
    new_count: wp.array[wp.int32]

    # Body transforms for world-space conversion
    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    # Per-prev claim word, packed (float_flip(dist_sq) << 32 | key_low32),
    # where key_low32 is the low 32 bits of the racing new contact's sort
    # key (deterministic per contact, invariant under non-deterministic
    # narrow-phase slot assignment -- see ``_pack_claim``).
    # Initialised to _CLAIM_SENTINEL each frame; race with atomic_min.
    prev_claim: wp.array[wp.int64]

    # Per-new candidate prev index (final value resolved in pass 2).
    match_index: wp.array[wp.int32]

    # Thresholds
    pos_threshold_sq: float
    normal_dot_threshold: float


@wp.kernel(enable_backward=False)
def _match_contacts_kernel(data: _MatchData):
    """Pass 1: pick each new contact's closest prev candidate and stake
    a packed claim on it via ``wp.atomic_min``.
    """
    tid = wp.tid()
    n_new = data.new_count[0]
    if tid >= n_new:
        data.match_index[tid] = MATCH_NOT_FOUND
        return

    n_old = data.prev_count[0]
    if n_old == 0:
        data.match_index[tid] = MATCH_NOT_FOUND
        return

    target_key = data.new_keys[tid]

    # World-space midpoint of the two contact points (symmetric in shape 0 /
    # shape 1).  Matches the quantity persisted by ``_save_sorted_state_kernel``
    # for the previous frame, so both sides of ``diff`` below measure the same
    # physical quantity.
    p0 = data.new_point0[tid]
    bid0 = data.shape_body[data.new_shape0[tid]]
    if bid0 == -1:
        p0w = p0
    else:
        p0w = wp.transform_point(data.body_q[bid0], p0)

    p1 = data.new_point1[tid]
    bid1 = data.shape_body[data.new_shape1[tid]]
    if bid1 == -1:
        p1w = p1
    else:
        p1w = wp.transform_point(data.body_q[bid1], p1)

    new_pos_w = 0.5 * (p0w + p1w)
    new_n = data.new_normal[tid]

    # Binary search the [range_lo, range_hi) interval of prev contacts
    # sharing the same (shape_a, shape_b) pair.  We ignore sort_sub_key
    # because for multi-contact manifolds (e.g. box-box face-face) the
    # sub-key assignment is not stable across frames; matching by the
    # exact key would spuriously break stable contacts.  Pair counts are
    # small (<= a few manifold points), so a linear scan inside the
    # range is cheap.
    pair_prefix = target_key & wp.int64(~0x7FFFFF)
    pair_end = pair_prefix + wp.int64(0x800000)  # 1 << 23
    range_lo = _lower_bound_int64(0, n_old, pair_prefix, data.prev_keys)
    range_hi = _lower_bound_int64(range_lo, n_old, pair_end, data.prev_keys)

    if range_lo >= range_hi:
        data.match_index[tid] = MATCH_NOT_FOUND
        return

    # Closest-point match within the pair range, gated by normal dot.
    best_idx = int(-1)
    best_dist_sq = float(data.pos_threshold_sq)
    for old_idx in range(range_lo, range_hi):
        old_pos = data.prev_pos_world[old_idx]
        diff = new_pos_w - old_pos
        dist_sq = wp.dot(diff, diff)
        old_n = data.prev_normal[old_idx]
        ndot = wp.dot(new_n, old_n)

        if dist_sq <= best_dist_sq and ndot >= data.normal_dot_threshold:
            best_dist_sq = dist_sq
            best_idx = old_idx

    if best_idx >= 0:
        data.match_index[tid] = wp.int32(best_idx)
        # Race for ownership of prev[best_idx] with a single atomic_min.
        # Closest distance wins; ties resolved by smallest sort_key low
        # 32 bits.  Using the sort key (instead of ``tid``) keeps the
        # winner invariant under the non-deterministic unsorted slot
        # assignment that the narrow phase gives us via ``wp.atomic_add``.
        wp.atomic_min(data.prev_claim, best_idx, _pack_claim(best_dist_sq, target_key))
    else:
        # Pair range exists but no contact within thresholds.
        data.match_index[tid] = MATCH_BROKEN


@wp.kernel(enable_backward=False)
def _clear_prev_claim_kernel(
    prev_claim: wp.array[wp.int64],
    prev_count: wp.array[wp.int32],
):
    """Reset only the active prefix of the claim buffer to ``_CLAIM_SENTINEL``.

    Launched with ``capacity`` threads so the per-frame launch fits a
    static CUDA graph, but each thread guards on ``prev_count[0]`` so we
    only touch the (typically much smaller) range of slots that ``match``
    will actually race on.  Slots beyond ``prev_count`` are never read
    by either kernel, so leaving them stale is safe.
    """
    i = wp.tid()
    if i < prev_count[0]:
        prev_claim[i] = _CLAIM_SENTINEL


@wp.kernel(enable_backward=False)
def _resolve_claims_kernel(
    match_index: wp.array[wp.int32],
    sort_keys: wp.array[wp.int64],
    prev_claim: wp.array[wp.int64],
    prev_was_matched: wp.array[wp.int32],
    new_count: wp.array[wp.int32],
    has_report: int,
):
    """Pass 2: keep winners, demote losers to :data:`MATCH_BROKEN`.

    The low 32 bits of ``prev_claim[cand]`` identify the winning
    contact by the low 32 bits of its sort key (deterministic per
    contact, invariant under unsorted slot reordering); everyone else
    who staked a claim on the same ``cand`` becomes :data:`MATCH_BROKEN`
    (no second-closest fallback).
    """
    tid = wp.tid()
    if tid >= new_count[0]:
        return

    cand = match_index[tid]
    if cand < wp.int32(0):
        return  # already MATCH_NOT_FOUND or MATCH_BROKEN

    winner_key_low = prev_claim[cand] & wp.int64(0xFFFFFFFF)
    my_key_low = sort_keys[tid] & wp.int64(0xFFFFFFFF)
    if winner_key_low == my_key_low:
        if has_report != 0:
            prev_was_matched[cand] = wp.int32(1)
    else:
        match_index[tid] = MATCH_BROKEN


# ------------------------------------------------------------------
# Save sorted state kernel
# ------------------------------------------------------------------


@wp.struct
class _SaveStateData:
    """Bundled arrays for the save-sorted-state kernel.

    ``src_point0`` / ``src_point1`` and their shape indices are consumed every
    frame to compute the symmetric world-space midpoint written into
    ``dst_pos_world``.  ``src_offset*`` and the ``dst_*_body`` columns are
    only consumed when ``has_sticky != 0``; when sticky is disabled the
    matcher passes dummy arrays for those slots and the kernel's
    ``if has_sticky`` guard skips the extra writes, so sticky-only columns
    need zero allocation in the non-sticky path.
    """

    src_keys: wp.array[wp.int64]
    src_point0: wp.array[wp.vec3]
    src_point1: wp.array[wp.vec3]
    src_offset0: wp.array[wp.vec3]
    src_offset1: wp.array[wp.vec3]
    src_shape0: wp.array[wp.int32]
    src_shape1: wp.array[wp.int32]
    src_normal: wp.array[wp.vec3]
    src_count: wp.array[wp.int32]

    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    dst_keys: wp.array[wp.int64]
    dst_pos_world: wp.array[wp.vec3]  # world-space midpoint of point0 and point1
    dst_normal: wp.array[wp.vec3]
    dst_point0_body: wp.array[wp.vec3]
    dst_point1_body: wp.array[wp.vec3]
    dst_offset0_body: wp.array[wp.vec3]
    dst_offset1_body: wp.array[wp.vec3]
    # Dedicated sticky-replay normal buffer.  Duplicates ``dst_normal`` content
    # but lives in its own allocation so sticky replay (which runs between
    # ``sort_full`` and the next ``save_sorted_state``) is not reading the
    # sorter's ``scratch_normal`` after the sort has clobbered it.
    dst_normal_sticky: wp.array[wp.vec3]
    dst_count: wp.array[wp.int32]

    has_sticky: int


@wp.kernel(enable_backward=False)
def _save_sorted_state_kernel(data: _SaveStateData):
    """Copy sorted contacts into the previous-frame buffers for next-frame matching.

    The persisted ``dst_pos_world`` is the world-space *midpoint* of the two
    contact points, so the next frame's match kernel compares a shape-symmetric
    quantity.
    """
    i = wp.tid()
    if i == 0:
        data.dst_count[0] = data.src_count[0]
    if i < data.src_count[0]:
        data.dst_keys[i] = data.src_keys[i]

        p0 = data.src_point0[i]
        bid0 = data.shape_body[data.src_shape0[i]]
        if bid0 == -1:
            p0w = p0
        else:
            p0w = wp.transform_point(data.body_q[bid0], p0)

        p1 = data.src_point1[i]
        bid1 = data.shape_body[data.src_shape1[i]]
        if bid1 == -1:
            p1w = p1
        else:
            p1w = wp.transform_point(data.body_q[bid1], p1)

        data.dst_pos_world[i] = 0.5 * (p0w + p1w)
        data.dst_normal[i] = data.src_normal[i]

        if data.has_sticky != 0:
            data.dst_point0_body[i] = p0
            data.dst_point1_body[i] = p1
            data.dst_offset0_body[i] = data.src_offset0[i]
            data.dst_offset1_body[i] = data.src_offset1[i]
            data.dst_normal_sticky[i] = data.src_normal[i]


# ------------------------------------------------------------------
# Sticky-mode replay (matched rows only)
# ------------------------------------------------------------------
#
# Sticky mode preserves only the fields that actually change across frames
# for a matched contact: the body-frame contact points (``point0``/``point1``)
# and offsets (``offset0``/``offset1``), plus the world-frame normal (which
# is already persisted for matching in ``prev_normal``, no extra allocation).
#
# Everything else is either key-derived or a per-shape constant that does
# not change between frames, so the new frame's values are already correct:
#
# - ``shape0`` / ``shape1``  : implied by the sort key; identical by
#   construction for matched contacts.
# - ``margin0`` / ``margin1``: ``radius_eff + margin``, per-shape constant.
# - ``stiffness`` / ``damping`` / ``friction``: per-shape constants, and
#   contact matching for hydroelastic contacts (the only path that writes
#   these) is not supported anyway.


@wp.struct
class _ReplayData:
    """Bundled arrays for the sticky replay kernel."""

    match_index: wp.array[wp.int32]
    contact_count: wp.array[wp.int32]

    prev_point0: wp.array[wp.vec3]
    prev_point1: wp.array[wp.vec3]
    prev_offset0: wp.array[wp.vec3]
    prev_offset1: wp.array[wp.vec3]
    prev_normal: wp.array[wp.vec3]

    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    offset0: wp.array[wp.vec3]
    offset1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]


@wp.kernel(enable_backward=False)
def _replay_matched_kernel(data: _ReplayData):
    tid = wp.tid()
    if tid >= data.contact_count[0]:
        return
    idx = data.match_index[tid]
    if idx < wp.int32(0):
        return  # MATCH_NOT_FOUND or MATCH_BROKEN -- keep new-frame data.
    data.point0[tid] = data.prev_point0[idx]
    data.point1[tid] = data.prev_point1[idx]
    data.offset0[tid] = data.prev_offset0[idx]
    data.offset1[tid] = data.prev_offset1[idx]
    data.normal[tid] = data.prev_normal[idx]


# ------------------------------------------------------------------
# Contact report kernels
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _collect_new_contacts_kernel(
    match_index: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    new_indices: wp.array[wp.int32],
    new_count: wp.array[wp.int32],
):
    """Collect indices of new or broken contacts (match_index < 0) after sorting."""
    i = wp.tid()
    if i >= contact_count[0]:
        return
    if match_index[i] < wp.int32(0):
        slot = wp.atomic_add(new_count, 0, wp.int32(1))
        new_indices[slot] = wp.int32(i)


@wp.kernel(enable_backward=False)
def _collect_broken_contacts_kernel(
    prev_was_matched: wp.array[wp.int32],
    prev_count: wp.array[wp.int32],
    broken_indices: wp.array[wp.int32],
    broken_count: wp.array[wp.int32],
):
    """Collect indices of old contacts that were not matched by any new contact."""
    i = wp.tid()
    if i >= prev_count[0]:
        return
    if prev_was_matched[i] == wp.int32(0):
        slot = wp.atomic_add(broken_count, 0, wp.int32(1))
        broken_indices[slot] = wp.int32(i)


# ------------------------------------------------------------------
# ContactMatcher class
# ------------------------------------------------------------------


class ContactMatcher:
    """Frame-to-frame contact matching using binary search on sorted keys.

    Internal helper owned by :class:`~newton.CollisionPipeline`.  All user-visible
    results (match index, new/broken index lists) are surfaced on the
    :class:`~newton.Contacts` container; this class only owns cross-frame state.

    Pre-allocates all buffers at construction time for CUDA graph capture
    compatibility.  See the module docstring for the per-frame call order and
    the ordering constraints between :meth:`match`, :meth:`replay_matched`,
    :meth:`build_report`, and :meth:`save_sorted_state`.

    Memory is minimised by reusing the sorter's existing scratch buffers for
    the previous-frame world-space contact midpoints and normals.  The matcher
    owns two small per-contact buffers in addition: the sorted key cache
    (8 bytes/contact) and the per-prev claim word used by the ``atomic_min``
    race that keeps new→prev injective (8 bytes/contact).  When
    ``contact_report`` is disabled, the ``prev_was_matched`` flag array is
    also skipped.

    .. note::
        Previous-frame state persists across :meth:`~newton.CollisionPipeline.collide`
        calls — that is the whole point.  In RL-style workflows where the user
        resets or teleports bodies between episodes, call :meth:`reset` after
        such discontinuities so the next frame starts fresh with all
        :data:`MATCH_NOT_FOUND`.

    Args:
        capacity: Maximum number of contacts (must match :class:`ContactSorter`).
        sorter: The :class:`ContactSorter` whose scratch buffers will be
            reused for storing previous-frame positions and normals.
        pos_threshold: World-space distance threshold [m] between the
            previous and current contact midpoints
            ``0.5 * (world(point0) + world(point1))``.  Contacts whose midpoint
            moved more than this between frames are considered broken.
        normal_dot_threshold: Minimum dot product between old and new contact
            normals.  Below this the contact is considered broken.
        contact_report: Allocate the ``prev_was_matched`` flag array needed
            to enumerate broken contacts in :meth:`build_report`.
        sticky: Allocate five extra per-contact ``wp.vec3`` buffers
            (``point0``/``point1``/``offset0``/``offset1`` body-frame, plus a
            dedicated ``normal`` buffer) used by :meth:`replay_matched`.  The
            world-frame normal needs its own allocation because sticky replay
            runs after ``ContactSorter.sort_full`` has clobbered the
            ``scratch_normal`` alias the match kernel reads pre-sort.  When
            ``False`` these attributes are ``None`` and no extra kernel
            launches are added.
        device: Device to allocate on.
    """

    def __init__(
        self,
        capacity: int,
        *,
        sorter: ContactSorter,
        pos_threshold: float = 0.0005,
        normal_dot_threshold: float = 0.995,
        contact_report: bool = False,
        sticky: bool = False,
        device: Devicelike = None,
    ):
        with wp.ScopedDevice(device):
            self._capacity = capacity
            self._pos_threshold_sq = pos_threshold * pos_threshold
            self._normal_dot_threshold = normal_dot_threshold
            self._sorter = sorter

            # Only buffer we must own: sorted keys survive across frames
            # (_sort_keys_copy is overwritten by _prepare_sort each frame).
            # Init with the sort-key sentinel so a debugger dump of the buffer
            # before the first save_sorted_state does not look like valid keys
            # for shape_a=0, shape_b=0, sub_key=0.
            self._prev_sorted_keys = wp.full(capacity, SORT_KEY_SENTINEL, dtype=wp.int64)
            self._prev_count = wp.zeros(1, dtype=wp.int32)

            # Per-prev claim word for the atomic_min race that keeps the
            # new→prev mapping injective (see module docstring).  Reset
            # to _CLAIM_SENTINEL each frame; the low 32 bits of the
            # surviving value identify the winning new contact by the low
            # 32 bits of its sort key (deterministic, invariant under
            # non-deterministic narrow-phase slot assignment -- see
            # ``_pack_claim``).
            self._prev_claim = wp.empty(capacity, dtype=wp.int64)

            # Contact report (optional).
            self._has_report = contact_report
            if contact_report:
                self._prev_was_matched = wp.zeros(capacity, dtype=wp.int32)
            else:
                # Dummy single-element array so the Warp struct is always valid.
                self._prev_was_matched = wp.zeros(1, dtype=wp.int32)

            # Sticky-mode buffers.  Only the body-frame point/offset pairs
            # and the world-frame normal need preserving -- shape indices,
            # margins, and per-shape properties are either key-derived or
            # per-shape constants and so identical on the next frame for a
            # matched contact.  The normal cannot reuse the sorter's
            # ``scratch_normal`` like the match kernel does, because sticky
            # replay runs *after* ``ContactSorter.sort_full`` and by then
            # ``scratch_normal`` has been clobbered with the current frame's
            # pre-sort normals by the sort's backup pass.
            self._sticky = sticky
            if sticky:
                self._prev_point0 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_point1 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_offset0 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_offset1 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_normal_sticky = wp.zeros(capacity, dtype=wp.vec3)
            else:
                self._prev_point0 = None
                self._prev_point1 = None
                self._prev_offset0 = None
                self._prev_offset1 = None
                self._prev_normal_sticky = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_report(self) -> bool:
        """Whether the contact report buffers are allocated."""
        return self._has_report

    @property
    def is_sticky(self) -> bool:
        """Whether sticky-mode full-record buffers are allocated."""
        return self._sticky

    @property
    def prev_contact_count(self) -> wp.array[wp.int32]:
        """Device-side previous frame contact count (single-element int32)."""
        return self._prev_count

    def reset(self) -> None:
        """Clear cross-frame state so the next frame starts fresh.

        Use this after any discontinuity that invalidates the previous
        frame's contacts (RL episode reset, teleported bodies, scene
        reload).  After ``reset()`` the next :meth:`match` produces all
        :data:`MATCH_NOT_FOUND` and :meth:`build_report` reports zero broken
        contacts.  Zeroing ``_prev_count`` is sufficient because both kernels
        gate on it.
        """
        self._prev_count.zero_()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def match(
        self,
        sort_keys: wp.array[wp.int64],
        contact_count: wp.array[wp.int32],
        point0: wp.array[wp.vec3],
        point1: wp.array[wp.vec3],
        shape0: wp.array[wp.int32],
        shape1: wp.array[wp.int32],
        normal: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        shape_body: wp.array[wp.int32],
        match_index_out: wp.array[wp.int32],
        *,
        device: Devicelike = None,
    ) -> None:
        """Match current unsorted contacts against last frame's sorted contacts.

        Must be called **before** :meth:`ContactSorter.sort_full`.

        Distance is measured between world-space contact midpoints
        (``0.5 * (world(point0) + world(point1))``) so the metric is symmetric
        in shape 0 / shape 1.

        Args:
            sort_keys: Current frame's unsorted int64 sort keys.
            contact_count: Single-element int array with active contact count.
            point0: Body-frame contact points on shape 0 (current frame).
            point1: Body-frame contact points on shape 1 (current frame).
            shape0: Shape indices for shape 0 (current frame).
            shape1: Shape indices for shape 1 (current frame).
            normal: Contact normals (current frame).
            body_q: Body transforms for the current frame.
            shape_body: Shape-to-body index map.
            match_index_out: Output int32 array to receive match results.
                Written directly (no intermediate copy).
            device: Device to launch on.
        """
        if self._has_report:
            self._prev_was_matched.zero_()

        # Reset only the active prefix of the claim buffer.  Launching
        # ``capacity`` threads keeps the call shape constant for graph
        # capture, but the kernel guards on ``prev_count`` so we touch
        # the minimum bytes — important for sparsely-loaded pipelines
        # where ``capacity >> prev_count``.
        wp.launch(
            _clear_prev_claim_kernel,
            dim=self._capacity,
            inputs=[self._prev_claim, self._prev_count],
            device=device,
        )

        data = _MatchData()
        data.prev_keys = self._prev_sorted_keys
        # Reuse sorter scratch buffers for prev-frame world-space data.
        data.prev_pos_world = self._sorter.scratch_pos_world
        data.prev_normal = self._sorter.scratch_normal
        data.prev_count = self._prev_count
        data.new_keys = sort_keys
        data.new_point0 = point0
        data.new_point1 = point1
        data.new_shape0 = shape0
        data.new_shape1 = shape1
        data.new_normal = normal
        data.new_count = contact_count
        data.body_q = body_q
        data.shape_body = shape_body
        data.match_index = match_index_out
        data.prev_claim = self._prev_claim
        data.pos_threshold_sq = self._pos_threshold_sq
        data.normal_dot_threshold = self._normal_dot_threshold

        wp.launch(_match_contacts_kernel, dim=self._capacity, inputs=[data], device=device)
        wp.launch(
            _resolve_claims_kernel,
            dim=self._capacity,
            inputs=[
                match_index_out,
                sort_keys,
                self._prev_claim,
                self._prev_was_matched,
                contact_count,
                1 if self._has_report else 0,
            ],
            device=device,
        )

    def save_sorted_state(
        self,
        sorted_keys: wp.array[wp.int64],
        contact_count: wp.array[wp.int32],
        sorted_point0: wp.array[wp.vec3],
        sorted_point1: wp.array[wp.vec3],
        sorted_shape0: wp.array[wp.int32],
        sorted_shape1: wp.array[wp.int32],
        sorted_normal: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        shape_body: wp.array[wp.int32],
        *,
        sorted_offset0: wp.array[wp.vec3] | None = None,
        sorted_offset1: wp.array[wp.vec3] | None = None,
        device: Devicelike = None,
    ) -> None:
        """Save current frame's sorted contacts for next-frame matching.

        Must be called **after** :meth:`ContactSorter.sort_full`.  The
        world-space midpoint of ``sorted_point0``/``sorted_point1`` and the
        sorted normal are written into the sorter's scratch buffers
        (:attr:`ContactSorter.scratch_pos_world` /
        :attr:`ContactSorter.scratch_normal`), which are idle between frames.

        When the matcher was built with ``sticky=True``, the body-frame
        point/offset columns are also persisted for :meth:`replay_matched` in
        the same kernel launch.  ``sorted_offset0`` / ``sorted_offset1`` are
        required in that case and ignored otherwise.

        Args:
            sorted_keys: Sorted int64 keys (use :attr:`ContactSorter.sorted_keys_view`).
            contact_count: Single-element int array with active contact count.
            sorted_point0: Sorted body-frame contact points on shape 0.
            sorted_point1: Sorted body-frame contact points on shape 1.
            sorted_shape0: Sorted shape 0 indices.
            sorted_shape1: Sorted shape 1 indices.
            sorted_normal: Sorted contact normals.
            body_q: Body transforms (current frame).
            shape_body: Shape-to-body index map.
            sorted_offset0, sorted_offset1: Required when sticky is enabled;
                ignored otherwise.
            device: Device to launch on.
        """
        data = _SaveStateData()
        data.src_keys = sorted_keys
        data.src_point0 = sorted_point0
        data.src_point1 = sorted_point1
        data.src_shape0 = sorted_shape0
        data.src_shape1 = sorted_shape1
        data.src_normal = sorted_normal
        data.src_count = contact_count
        data.body_q = body_q
        data.shape_body = shape_body
        data.dst_keys = self._prev_sorted_keys
        # Write world-space midpoint and normal into the sorter's scratch buffers.
        data.dst_pos_world = self._sorter.scratch_pos_world
        data.dst_normal = self._sorter.scratch_normal
        data.dst_count = self._prev_count

        if self._sticky:
            if sorted_offset0 is None or sorted_offset1 is None:
                raise ValueError("save_sorted_state requires sorted_offset0/offset1 when sticky is enabled")
            data.src_offset0 = sorted_offset0
            data.src_offset1 = sorted_offset1
            data.dst_point0_body = self._prev_point0
            data.dst_point1_body = self._prev_point1
            data.dst_offset0_body = self._prev_offset0
            data.dst_offset1_body = self._prev_offset1
            data.dst_normal_sticky = self._prev_normal_sticky
            data.has_sticky = 1
        else:
            # The struct requires a valid array for every field -- the
            # kernel guards with has_sticky==0 and never reads/writes them.
            data.src_offset0 = sorted_point0
            data.src_offset1 = sorted_point0
            data.dst_point0_body = self._sorter.scratch_pos_world
            data.dst_point1_body = self._sorter.scratch_pos_world
            data.dst_offset0_body = self._sorter.scratch_pos_world
            data.dst_offset1_body = self._sorter.scratch_pos_world
            data.dst_normal_sticky = self._sorter.scratch_pos_world
            data.has_sticky = 0

        wp.launch(_save_sorted_state_kernel, dim=self._capacity, inputs=[data], device=device)

    def replay_matched(
        self,
        contact_count: wp.array[wp.int32],
        match_index: wp.array[wp.int32],
        *,
        point0: wp.array[wp.vec3],
        point1: wp.array[wp.vec3],
        offset0: wp.array[wp.vec3],
        offset1: wp.array[wp.vec3],
        normal: wp.array[wp.vec3],
        device: Devicelike = None,
    ) -> None:
        """Overwrite matched rows with the saved previous-frame contact geometry.

        Only valid when the matcher was constructed with ``sticky=True``.  Must
        run **after** :meth:`ContactSorter.sort_full` and **before**
        :meth:`save_sorted_state`.  Unmatched rows (``match_index < 0``) are
        left untouched so new contacts keep their fresh narrow-phase geometry.
        Only ``point0``/``point1``/``offset0``/``offset1``/``normal`` are
        restored; other fields (``shape0``/``shape1``, margins, ...) are
        already identical for a matched contact.

        Args:
            contact_count: Single-element int array with the active contact count.
            match_index: Sorted match_index array (from :class:`Contacts`).
            point0, point1, offset0, offset1, normal: Current-frame sorted
                contact record to be overwritten on matched rows.
            device: Device to launch on.
        """
        if not self._sticky:
            raise ValueError("replay_matched requires the matcher to be constructed with sticky=True")

        data = _ReplayData()
        data.match_index = match_index
        data.contact_count = contact_count
        data.prev_point0 = self._prev_point0
        data.prev_point1 = self._prev_point1
        data.prev_offset0 = self._prev_offset0
        data.prev_offset1 = self._prev_offset1
        # Use the dedicated sticky normal buffer, NOT sorter.scratch_normal:
        # replay runs after ``sort_full``, which has clobbered scratch_normal
        # with the current frame's pre-sort normals during its backup pass.
        data.prev_normal = self._prev_normal_sticky
        data.point0 = point0
        data.point1 = point1
        data.offset0 = offset0
        data.offset1 = offset1
        data.normal = normal

        wp.launch(_replay_matched_kernel, dim=self._capacity, inputs=[data], device=device)

    def build_report(
        self,
        match_index: wp.array[wp.int32],
        contact_count: wp.array[wp.int32],
        new_indices: wp.array[wp.int32],
        new_count: wp.array[wp.int32],
        broken_indices: wp.array[wp.int32],
        broken_count: wp.array[wp.int32],
        *,
        device: Devicelike = None,
    ) -> None:
        """Build new/broken contact index lists (optional, post-sort).

        Must be called **after** :meth:`ContactSorter.sort_full` and **before**
        :meth:`save_sorted_state` (``save_sorted_state`` overwrites
        ``_prev_count``, which this method reads to bound the broken-contact
        enumeration).

        After this call, ``new_indices`` / ``new_count`` hold indices of
        contacts in the current sorted buffer that have no prior match
        (``match_index < 0``), and ``broken_indices`` / ``broken_count`` hold
        indices of old contacts that were not matched by any new contact.

        Args:
            match_index: Sorted match_index array (from :class:`Contacts`).
            contact_count: Single-element int array with active contact count.
            new_indices: Output array to receive new-contact indices.
            new_count: Single-element output counter for new contacts.
            broken_indices: Output array to receive broken-contact indices
                (indexing the previous frame's sorted buffer).
            broken_count: Single-element output counter for broken contacts.
            device: Device to launch on.
        """
        if not self._has_report:
            return

        new_count.zero_()
        broken_count.zero_()

        wp.launch(
            _collect_new_contacts_kernel,
            dim=self._capacity,
            inputs=[match_index, contact_count, new_indices, new_count],
            device=device,
        )
        wp.launch(
            _collect_broken_contacts_kernel,
            dim=self._capacity,
            inputs=[self._prev_was_matched, self._prev_count, broken_indices, broken_count],
            device=device,
        )
