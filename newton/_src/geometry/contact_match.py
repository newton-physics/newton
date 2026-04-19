# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame-to-frame contact matching via binary search on sorted contact keys.

Given the previous frame's sorted contacts (keys, world-space positions,
normals) and the current frame's unsorted contacts, this module finds
correspondences using the deterministic sort key from
:func:`~newton._src.geometry.contact_data.make_contact_sort_key`.

For each new contact the matcher binary-searches the previous frame's
sorted keys for the ``(shape_a, shape_b)`` pair range — ignoring the
``sort_sub_key`` bits — then picks the closest previous contact in that
range whose normal also passes the dot-product threshold.  The result
is a per-contact match index:

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

    claim = (float_flip(dist_sq) << 32) | tid

``float_flip`` reinterprets the non-negative ``dist_sq`` as a
sortable ``uint32``, so the high 32 bits order claims by ascending
distance; the low 32 bits hold the new contact index, breaking ties
deterministically (smallest ``tid`` wins).  After the match kernel
runs, a small finalize kernel reads ``prev_claim[best_idx]`` and
demotes any new contact whose ``tid`` does not appear in the low bits
to :data:`MATCH_BROKEN`.  Losers are *not* re-matched against a
second-closest prev (kept for simplicity and speed).

Cost: one ``int64[capacity]`` buffer, one ``wp.atomic_min`` per new
contact, and one short finalize kernel launch.  No ``atomic_cas``.

Memory efficiency
-----------------
The matcher reuses the :class:`ContactSorter`'s existing scratch buffers
(:attr:`ContactSorter.scratch_pos_world`, :attr:`ContactSorter.scratch_normal`)
to store previous-frame world-space positions and normals between frames.
This works because matching runs *before* sorting each frame (so the
scratch data is still intact), and the save step runs *after* sorting
(overwriting the scratch with the new sorted data).  The only additional
per-contact allocation is the ``_prev_sorted_keys`` buffer (8 bytes/contact)
since the sorter's key buffer is overwritten by ``_prepare_sort`` each frame.

Per-frame call order (inside :class:`~newton.CollisionPipeline`)::

    matcher.match(...)  # before ContactSorter.sort_full()
    sorter.sort_full(...)  # match_index is permuted with contacts
    matcher.build_report(...)  # optional; must precede save_sorted_state
    matcher.save_sorted_state(...)  # after sorting and report

The ``build_report`` / ``save_sorted_state`` ordering is load-bearing:
``save_sorted_state`` overwrites ``_prev_count`` with the current frame's
count, while ``build_report`` reads the *old* ``_prev_count`` to bound the
broken-contact enumeration.  Swapping the two silently narrows the
broken-contact scan to the current frame's active range.
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
# any packed (flipped_dist << 32 | tid) any kernel will ever produce,
# so the first ``atomic_min`` always wins.
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
def _pack_claim(dist_sq: float, tid: int) -> wp.int64:
    """Pack ``(dist_sq, tid)`` into a single int64 for ``atomic_min``.

    High 32 bits: ``float_flip(dist_sq)`` — ascending by distance.
    Low 32 bits:  ``tid`` — deterministic tie-break (smallest wins).
    """
    flipped = wp.int64(_float_flip(dist_sq))
    return (flipped << wp.int64(32)) | wp.int64(tid)


# ------------------------------------------------------------------
# Match kernel
# ------------------------------------------------------------------


@wp.struct
class _MatchData:
    """Bundled arrays for the contact match kernel."""

    # Previous frame (sorted) — pos/normal reuse ContactSorter scratch buffers
    prev_keys: wp.array[wp.int64]
    prev_pos_world: wp.array[wp.vec3]
    prev_normal: wp.array[wp.vec3]
    prev_count: wp.array[wp.int32]

    # Current frame (unsorted)
    new_keys: wp.array[wp.int64]
    new_point0: wp.array[wp.vec3]
    new_shape0: wp.array[wp.int32]
    new_normal: wp.array[wp.vec3]
    new_count: wp.array[wp.int32]

    # Body transforms for world-space conversion
    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    # Per-prev claim word, packed (float_flip(dist_sq) << 32 | tid).
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

    # Compute world-space position for the new contact.
    bid = data.shape_body[data.new_shape0[tid]]
    if bid == -1:
        new_pos_w = data.new_point0[tid]
    else:
        new_pos_w = wp.transform_point(data.body_q[bid], data.new_point0[tid])
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
        # Closest distance wins; ties resolved by lowest tid.
        wp.atomic_min(data.prev_claim, best_idx, _pack_claim(best_dist_sq, tid))
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
    prev_claim: wp.array[wp.int64],
    prev_was_matched: wp.array[wp.int32],
    new_count: wp.array[wp.int32],
    has_report: int,
):
    """Pass 2: keep winners, demote losers to :data:`MATCH_BROKEN`.

    The low 32 bits of ``prev_claim[cand]`` identify the winning
    ``tid``; everyone else who staked a claim on the same ``cand``
    becomes :data:`MATCH_BROKEN` (no second-closest fallback).
    """
    tid = wp.tid()
    if tid >= new_count[0]:
        return

    cand = match_index[tid]
    if cand < wp.int32(0):
        return  # already MATCH_NOT_FOUND or MATCH_BROKEN

    winner_tid = wp.int32(prev_claim[cand] & wp.int64(0xFFFFFFFF))
    if winner_tid == wp.int32(tid):
        if has_report != 0:
            prev_was_matched[cand] = wp.int32(1)
    else:
        match_index[tid] = MATCH_BROKEN


# ------------------------------------------------------------------
# Save sorted state kernel
# ------------------------------------------------------------------


@wp.struct
class _SaveStateData:
    """Bundled arrays for the save-sorted-state kernel."""

    src_keys: wp.array[wp.int64]
    src_point0: wp.array[wp.vec3]
    src_shape0: wp.array[wp.int32]
    src_normal: wp.array[wp.vec3]
    src_count: wp.array[wp.int32]

    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    dst_keys: wp.array[wp.int64]
    dst_pos_world: wp.array[wp.vec3]
    dst_normal: wp.array[wp.vec3]
    dst_count: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def _save_sorted_state_kernel(data: _SaveStateData):
    """Copy sorted contacts into the previous-frame buffers for next-frame matching."""
    i = wp.tid()
    if i == 0:
        data.dst_count[0] = data.src_count[0]
    if i < data.src_count[0]:
        data.dst_keys[i] = data.src_keys[i]
        bid = data.shape_body[data.src_shape0[i]]
        if bid == -1:
            data.dst_pos_world[i] = data.src_point0[i]
        else:
            data.dst_pos_world[i] = wp.transform_point(data.body_q[bid], data.src_point0[i])
        data.dst_normal[i] = data.src_normal[i]


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
    compatibility.  The typical per-frame call sequence is::

        matcher.match(...)  # before ContactSorter.sort_full()
        sorter.sort_full(...)  # match_index is permuted with contacts
        matcher.build_report(...)  # optional; must precede save_sorted_state
        matcher.save_sorted_state(...)  # after sorting and report

    The ``build_report`` → ``save_sorted_state`` order is required:
    ``save_sorted_state`` overwrites ``_prev_count`` with the current frame's
    count, while ``build_report`` reads the *old* ``_prev_count`` to bound
    the broken-contact enumeration.

    Memory is minimised by reusing the *sorter*'s existing scratch buffers
    for the previous-frame world-space positions and normals.  The
    matcher owns two small per-contact buffers in addition: the sorted
    key cache (8 bytes/contact) and the per-prev claim word used by the
    ``atomic_min`` race that keeps new→prev injective (8 bytes/contact).
    When ``contact_report`` is disabled, the ``prev_was_matched`` flag
    array (4 bytes) is also skipped.

    .. note::
        Previous-frame state persists across :meth:`~newton.CollisionPipeline.collide`
        calls — that is the whole point.  But in RL-style workflows where a
        user resets or teleports all bodies between episodes, the stale
        previous-frame data will produce spurious matches on the next frame.
        Call :meth:`reset` after such discontinuities to zero ``_prev_count``
        so the next frame starts fresh with all ``MATCH_NOT_FOUND``.

    Args:
        capacity: Maximum number of contacts (must match :class:`ContactSorter`).
        sorter: The :class:`ContactSorter` whose scratch buffers will be
            reused for storing previous-frame positions and normals.
        pos_threshold: World-space distance threshold [m].  Contacts whose
            positions moved more than this between frames are considered broken.
        normal_dot_threshold: Minimum dot product between old and new contact
            normals.  Below this the contact is considered broken.
        contact_report: Allocate the ``prev_was_matched`` flag array needed
            to enumerate broken contacts in :meth:`build_report`.
        device: Device to allocate on.
    """

    def __init__(
        self,
        capacity: int,
        *,
        sorter: ContactSorter,
        pos_threshold: float = 0.02,
        normal_dot_threshold: float = 0.9,
        contact_report: bool = False,
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
            # surviving value identify the winning new contact ``tid``.
            self._prev_claim = wp.empty(capacity, dtype=wp.int64)

            # Contact report (optional).
            self._has_report = contact_report
            if contact_report:
                self._prev_was_matched = wp.zeros(capacity, dtype=wp.int32)
            else:
                # Dummy single-element array so the Warp struct is always valid.
                self._prev_was_matched = wp.zeros(1, dtype=wp.int32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_report(self) -> bool:
        """Whether the contact report buffers are allocated."""
        return self._has_report

    @property
    def prev_contact_count(self) -> wp.array:
        """Device-side previous frame contact count (single-element int32)."""
        return self._prev_count

    def reset(self) -> None:
        """Clear cross-frame state so the next frame starts fresh.

        Use this after any discontinuity that invalidates the previous
        frame's contacts — e.g. resetting an RL environment, teleporting
        bodies, or loading a new scene.  After ``reset()`` the next call to
        :meth:`match` produces all :data:`MATCH_NOT_FOUND` and
        :meth:`build_report` reports zero broken contacts.

        Zeroing ``_prev_count`` is sufficient to short-circuit both paths;
        the previous-frame key and position buffers are not touched, but
        they will not be read because both kernels gate on ``_prev_count``
        / ``prev_count``.
        """
        self._prev_count.zero_()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def match(
        self,
        sort_keys: wp.array,
        contact_count: wp.array,
        point0: wp.array,
        shape0: wp.array,
        normal: wp.array,
        body_q: wp.array,
        shape_body: wp.array,
        match_index_out: wp.array,
        *,
        device: Devicelike = None,
    ) -> None:
        """Match current unsorted contacts against last frame's sorted contacts.

        Must be called **before** :meth:`ContactSorter.sort_full`.

        Args:
            sort_keys: Current frame's unsorted int64 sort keys.
            contact_count: Single-element int array with active contact count.
            point0: Body-frame contact points on shape 0 (current frame).
            shape0: Shape indices for shape 0 (current frame).
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
        data.new_shape0 = shape0
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
                self._prev_claim,
                self._prev_was_matched,
                contact_count,
                1 if self._has_report else 0,
            ],
            device=device,
        )

    def save_sorted_state(
        self,
        sorted_keys: wp.array,
        contact_count: wp.array,
        sorted_point0: wp.array,
        sorted_shape0: wp.array,
        sorted_normal: wp.array,
        body_q: wp.array,
        shape_body: wp.array,
        *,
        device: Devicelike = None,
    ) -> None:
        """Save current frame's sorted contacts for next-frame matching.

        Must be called **after** :meth:`ContactSorter.sort_full`.

        World-space positions and normals are written into the sorter's scratch
        buffers (:attr:`ContactSorter.scratch_pos_world`,
        :attr:`ContactSorter.scratch_normal`), which are idle between frames.
        Sorted keys go to an owned buffer since the sorter's key buffer is
        overwritten by ``_prepare_sort`` each frame.

        Args:
            sorted_keys: Sorted int64 keys (use :attr:`ContactSorter.sorted_keys_view`).
            contact_count: Single-element int array with active contact count.
            sorted_point0: Sorted body-frame contact points on shape 0.
            sorted_shape0: Sorted shape 0 indices.
            sorted_normal: Sorted contact normals.
            body_q: Body transforms (current frame).
            shape_body: Shape-to-body index map.
            device: Device to launch on.
        """
        data = _SaveStateData()
        data.src_keys = sorted_keys
        data.src_point0 = sorted_point0
        data.src_shape0 = sorted_shape0
        data.src_normal = sorted_normal
        data.src_count = contact_count
        data.body_q = body_q
        data.shape_body = shape_body
        data.dst_keys = self._prev_sorted_keys
        # Write world-space positions and normals into sorter's scratch buffers.
        data.dst_pos_world = self._sorter.scratch_pos_world
        data.dst_normal = self._sorter.scratch_normal
        data.dst_count = self._prev_count

        wp.launch(_save_sorted_state_kernel, dim=self._capacity, inputs=[data], device=device)

    def build_report(
        self,
        match_index: wp.array,
        contact_count: wp.array,
        new_indices: wp.array,
        new_count: wp.array,
        broken_indices: wp.array,
        broken_count: wp.array,
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
