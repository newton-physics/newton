# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-pair contact reduction: reduction groups and table sizing for the global reducer's group mode.

Multi-shape bodies multiply narrow-phase output: a foot approximated by 7
cylinders emits up to 28 candidate contacts against a plane and up to 49
against another such foot, even when the underlying physics is one flat patch.
Body-pair reduction represents that patch with one depth slot and two families of six
sampled footprint slots rather than treating collider decomposition as
physical detail.

**Approximation, stated precisely.** For contacts RETAINED at the true hull, an
interior point's normal force is a convex combination of hull-point normal
forces (support/tipping statics preserved) and translational friction capacity
``sum mu*N_i`` is position-independent.  But the hull itself is SAMPLED by
six fixed scan directions, and the sampling error
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

The reduction itself is the global contact reducer's group mode (see
:class:`~newton._src.geometry.contact_reduction_global.GroupReductionConfig`):
every narrow-phase contact is staged, grouped by ``(group0, group1, normal bin,
spatial cell)`` where a group is a body plus an exact material signature (or a
static shape on its own), and only the winners reach the ``Contacts`` buffer.
Each group keeps its deepest contact, the extremes of all its contacts along
six scan directions, and the extremes of its touching contacts (separation
``<= 0``) along the same directions. Contacts discarded this way are typically
interior points whose force is a convex combination of the hull points', though
a sparsely populated hull can also lose true vertices that fall between scan
directions (see above). This module builds the groups and sizes the group table.

**When this pays.** Only bodies whose collision is decomposed into several
primitives generate more candidates per pair than the slot budget. Dense
multi-collider patches can repay the extra staging and registration work by
substantially reducing solver rows; scenes dominated by one small manifold per
body pair cannot and pay pure overhead. It is opt-in for that reason.
``stats()`` reports paired input/kept totals; a kept/input ratio near one means
this scene does not want it. Benchmark the complete collision-plus-solver step on the target GPU
rather than inferring speedup from contact count alone.

Depth is ranked with the canonical signed effective-surface separation
``dot(normal, point1 - point0) - margin0 - margin1``.  It picks the depth
representative and its sign gates the touching slot family; it never biases
the ordering inside a spatial slot.  The reduction still has an explicit
spatial-cell length scale, and no depth classification can leave a patch
without footprint support.

Reduction never crosses a normal bin, so multi-patch configurations (a body
touching floor and wall at once) keep representatives of every patch.  On
hashtable probe exhaustion the reduction fails open: if any contact cannot be
registered within the bounded probe budget, the frame keeps every contact.  Probe
exhaustion can mean a full table or a long collision cluster; telemetry does
not claim to distinguish the two.

**Memoryless selection.** The kept set is a function of the current contact
geometry only.  On curved or mesh support (flat analytic planes are immune) the
score differences between a patch's near-symmetric contacts are curvature-sized,
so the winners can hand off between steps.

**XPBD is not a supported consumer.** XPBD predicts body poses before deciding
which speculative candidates are active, while the reduction selects footprint
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
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..sim.model import Model
from .contact_reduction_global import (
    GROUP_STAT_CELL_CLAMP_EVENTS,
    GROUP_STAT_FALLBACK_FRAMES,
    GROUP_STAT_INPUT_OVERFLOW_FRAMES,
    GROUP_STAT_MAX_CONTACTS_IN,
    GROUP_STAT_MAX_CONTACTS_KEPT,
    GROUP_STAT_MAX_ENTRIES,
    GROUP_STAT_OUTRANKED_DISCARDS,
    GROUP_STAT_PROBE_FAILURES,
    GROUP_STAT_SUM_CONTACTS_IN,
    GROUP_STAT_SUM_CONTACTS_KEPT,
    GROUP_STAT_TOTAL_FRAMES,
    GROUP_VALUES_PER_KEY,
    MAX_GROUP_ID,
    GlobalContactReducer,
    GroupReductionConfig,
)


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
            # Model.add_attribute() leaves the assignment unset for model-owned attributes.
            if spec.frequency == Model.AttributeFrequency.SHAPE
            and spec.assignment in (None, Model.AttributeAssignment.MODEL)
            and name not in Model._CORE_ATTRIBUTE_SPECS
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
    unordered way :func:`_make_group_key` packs it.  The caller passes an
    explicit pair list; the model's default pairs, which SAP and NXN use, are
    bounded by :func:`build_reduction_group_pair_upper_bound` without being
    enumerated.

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


def build_reduction_group_pair_upper_bound(shape_body, shape_world, shape_active, shape_group) -> int:
    """Bound the distinct group pairs the model's default contact pairs can produce, in O(shape count).

    Default pairs join shapes on different bodies in one world, or with a
    global shape, and never two static shapes.  A dynamic group lives on one
    body and a static group is one shape, so counting group pairs under the
    same rules bounds :func:`build_reduction_group_pair_bound` of those pairs.

    Args:
        shape_body: Per-shape body index, ``-1`` for static shapes.
        shape_world: Per-shape world index, ``-1`` for global shapes.
        shape_active: Per-shape mask of shapes that take part in contact pairs.
        shape_group: Per-shape group ids from :func:`build_reduction_groups`.

    Returns:
        An upper bound on the number of distinct unordered group pairs.
    """
    active = np.asarray(shape_active, dtype=bool)
    groups = np.asarray(shape_group, dtype=np.int64)[active]
    if groups.size == 0:
        return 0
    worlds = np.asarray(shape_world, dtype=np.int64)[active] + 1
    slots = int(worlds.max()) + 1
    # Count a group once per world it has shapes in.
    _, first = np.unique(groups * slots + worlds, return_index=True)
    group_world = worlds[first]
    group_dynamic = np.asarray(shape_body, dtype=np.int64)[active][first] >= 0
    # Slot 0 holds global groups, slot w + 1 the groups of world w.
    dynamic = np.bincount(group_world[group_dynamic], minlength=slots).astype(np.int64)
    static = np.bincount(group_world[~group_dynamic], minlength=slots).astype(np.int64)
    dg, sg = int(dynamic[0]), int(static[0])
    dw, sw = dynamic[1:], static[1:]
    per_world = dw * (dw - 1) // 2 + dw * (sw + dg + sg) + sw * dg
    return int(per_world.sum()) + dg * (dg - 1) // 2 + dg * sg


def build_group_reduction_config(
    model,
    cell_size: float,
    hashtable_headroom: float,
    material_attributes,
    shape_pairs,
    rigid_contact_max: int,
    has_mesh_like: bool,
) -> tuple[GroupReductionConfig, int, int]:
    """Build the group-reduction settings for a pipeline.

    Args:
        model: The simulation model.
        cell_size: Spatial cell edge [m].
        hashtable_headroom: Multiplier on the derived group-table capacity.
        material_attributes: Per-shape attributes that split groups (see :func:`build_reduction_groups`).
        shape_pairs: The explicit broad-phase pair list, or ``None`` for the model's default pairs.
        rigid_contact_max: Staging capacity; also caps the table, since a group needs a contact.
        has_mesh_like: Whether the scene has mesh or heightfield shapes.

    Returns:
        ``(config, group_pair_bound, hashtable_capacity_request)``.
    """
    # Imported here: shape_contact_pairs imports the geometry package.
    from ..sim.shape_contact_pairs import (  # noqa: PLC0415
        _shape_contact_pair_counts,
        _shape_contact_pair_override,
        _shape_contact_pair_topology,
        _shape_contact_pairs_for_mask,
    )

    shape_group, group_count = build_reduction_groups(model, material_attributes)
    if group_count > MAX_GROUP_ID + 1:
        raise ValueError(f"body_pair_reduction supports at most {MAX_GROUP_ID + 1} groups, got {group_count}")
    # Size the group table from the broad phase's own candidate pairs; enumerate only an explicit list.
    if shape_pairs is not None or _shape_contact_pair_override(model)[0]:
        candidate_pairs = _shape_contact_pairs_for_mask(
            model, np.ones(model.shape_count, dtype=bool), shape_pairs=shape_pairs
        )
        group_pair_bound = build_reduction_group_pair_bound(candidate_pairs, shape_group)
    else:
        topology = _shape_contact_pair_topology(model)
        group_pair_bound = min(
            int(_shape_contact_pair_counts(model).sum()),
            build_reduction_group_pair_upper_bound(
                topology.shape_body, topology.shape_world, topology._active(), shape_group
            ),
        )
    # A replicated scene can have one reachable pair per body, yet each pair spans
    # several bins and cells: reserve four patch entries per group at half load.
    group_budget = 8 * len(np.unique(shape_group))
    # A curved mesh or heightfield patch spreads one pair over many bins and cells,
    # which the topology cannot see, so also reserve one entry per eight staged contacts.
    staged_budget = rigid_contact_max // 8 if has_mesh_like else 0
    capacity_request = int(max(group_pair_bound, group_budget, staged_budget) * float(hashtable_headroom))
    capacity_request = max(1024, min(capacity_request, rigid_contact_max))
    config = GroupReductionConfig(
        shape_group=shape_group,
        cell_size=float(np.float32(cell_size)),
        hashtable_capacity=capacity_request,
        up_rotation=_up_axis_rotation(int(model.up_axis)),
    )
    return config, group_pair_bound, capacity_request


class BodyPairReduction:
    """Telemetry and group refresh for a pipeline's group-mode global reducer."""

    def __init__(
        self, reducer: GlobalContactReducer, rigid_contact_max: int, group_pair_bound: int, capacity_request: int
    ):
        self.reducer = reducer
        self.rigid_contact_max = rigid_contact_max
        self.group_pair_bound = group_pair_bound
        self.hashtable_capacity_request = capacity_request

    @property
    def shape_group(self) -> wp.array:
        """Per-shape reduction-group ids on the device."""
        return self.reducer.shape_group

    @property
    def verify(self) -> bool:
        """Count discarded contacts that out-rank a recorded winner."""
        return self.reducer.verify

    @verify.setter
    def verify(self, value: bool):
        self.reducer.verify = bool(value)

    def stats(self) -> dict:
        """Whole-run reduction telemetry (forces a device sync; read outside capture).

        Returns:
            ``probe_failures``: group-table insertions that ran out of probes; each one
            makes its whole frame keep every contact (counted in ``fallback_frames``).
            ``invariant_violations``: discarded contacts that out-rank a recorded
            winner, counted only with :attr:`verify` on.
            ``cell_clamp_events``: contacts whose spatial cell hit the packed +/-31 range.
            ``max_contacts_in`` / ``max_contacts_kept``: peak admitted and kept contacts per
            frame. ``max_contacts_in`` is bounded by the staging capacity, so it does not
            estimate the capacity a scene needs after an overflow.
            ``total_frames``, ``sum_contacts_in``, ``sum_contacts_kept``: whole-run totals;
            ``sum_contacts_kept / sum_contacts_in`` is the achieved reduction ratio.
            ``max_hashtable_entries`` / ``hashtable_capacity`` / ``hashtable_load``: peak
            distinct (group pair, bin, cell) entries against the table.
            ``input_overflow_frames``: frames whose contacts overflowed the staging buffer.
        """
        self._raise_if_capturing("stats()")
        v = self.reducer.group_stats.numpy()
        entries = int(v[GROUP_STAT_MAX_ENTRIES])
        capacity = self.reducer.hashtable.capacity
        return {
            "invariant_violations": int(v[GROUP_STAT_OUTRANKED_DISCARDS]),
            "probe_failures": int(v[GROUP_STAT_PROBE_FAILURES]),
            "cell_clamp_events": int(v[GROUP_STAT_CELL_CLAMP_EVENTS]),
            "max_contacts_in": int(v[GROUP_STAT_MAX_CONTACTS_IN]),
            "max_contacts_kept": int(v[GROUP_STAT_MAX_CONTACTS_KEPT]),
            "max_hashtable_entries": entries,
            "hashtable_capacity": capacity,
            "hashtable_load": entries / capacity,
            "input_overflow_frames": int(v[GROUP_STAT_INPUT_OVERFLOW_FRAMES]),
            "fallback_frames": int(v[GROUP_STAT_FALLBACK_FRAMES]),
            "total_frames": int(v[GROUP_STAT_TOTAL_FRAMES]),
            "sum_contacts_in": int(v[GROUP_STAT_SUM_CONTACTS_IN]),
            "sum_contacts_kept": int(v[GROUP_STAT_SUM_CONTACTS_KEPT]),
        }

    def clear_stats(self):
        """Zero the whole-run telemetry (host-side, outside capture)."""
        self._raise_if_capturing("clear_stats()")
        self.reducer.group_stats.zero_()

    def describe(self) -> dict:
        """Sizes fixed at construction."""
        capacity = self.reducer.hashtable.capacity
        return {
            "rigid_contact_max": self.rigid_contact_max,
            "hashtable_capacity": capacity,
            "group_pair_bound": self.group_pair_bound,
            "hashtable_capacity_request": self.hashtable_capacity_request,
            "slots_per_entry": GROUP_VALUES_PER_KEY,
            "slot_value_bytes": capacity * GROUP_VALUES_PER_KEY * 8,
        }

    def _raise_if_capturing(self, what: str):
        """Reject host-side telemetry during CUDA graph capture, where it would sync or be recorded."""
        device = self.reducer.hashtable.keys.device
        if device.is_cuda and device.is_capturing:
            raise RuntimeError(f"{what} is host-side telemetry; call it outside CUDA graph capture")
