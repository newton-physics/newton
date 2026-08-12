# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile collision filtering between MuJoCo and Newton representations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MUJOCO_COLLISION_MASK_UNSET = np.iinfo(np.int64).min
"""Sentinel for Newton shapes without preserved MuJoCo collision masks."""

MUJOCO_COLLISION_MASK_DOMAIN_UNSET = -1
"""Sentinel for shapes whose collision masks have no MJCF import provenance."""

NEWTON_COLLISION_MASK_MAX_SHAPE_COUNT = 256
NEWTON_COLLISION_MASK_MAX_EXCLUDED_PAIR_COUNT = 1024


def mujoco_mask_to_signed(value: int) -> int:
    """Return a 32-bit MuJoCo mask as its signed integer representation."""
    normalized = np.asarray(int(value) & 0xFFFFFFFF, dtype=np.uint32)
    return int(normalized.view(np.int32))


@dataclass(frozen=True)
class CollisionMaskCompileResult:
    """Result of compiling collision masks into Newton collision primitives."""

    groups: np.ndarray
    """Normalized collision group per shape, shape ``[shape_count]``."""

    excluded_pairs: np.ndarray
    """Additional canonical exclusion pairs, shape ``[pair_count, 2]``."""

    class_count: int
    """Number of distinct ``(contype, conaffinity)`` classes."""

    group_count: int
    """Number of nonzero collision groups used by the result."""

    search_nodes: int
    """Number of branch-and-bound states visited."""

    optimal: bool
    """Whether the search proved optimality within the mask-class model."""


@dataclass(frozen=True)
class CollisionGraphCompileResult:
    """Result of compiling a Newton collision graph into MuJoCo masks."""

    collision_type: np.ndarray
    """MuJoCo ``contype`` values, shape ``[shape_count]``."""

    collision_affinity: np.ndarray
    """MuJoCo ``conaffinity`` values, shape ``[shape_count]``."""

    bit_count: int
    """Number of mask bits used, or the first count beyond the capacity."""

    exact: bool
    """Whether the masks exactly represent every Newton shape pair."""

    uncovered_pair_count: int | None
    """Number of uncovered Newton pairs, or ``None`` when compilation was skipped."""

    skipped: bool
    """Whether compilation was skipped before constructing the pair graph."""


def _group_pair_allowed(group_a: int, group_b: int) -> bool:
    """Return whether two Newton collision groups interact."""
    if group_a == 0 or group_b == 0:
        return False
    if group_a > 0:
        return group_a == group_b or group_b < 0
    return group_a != group_b


def _normalize_masks(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    """Normalize integer masks to MuJoCo's 32-bit bitwise domain."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        return np.empty(0, dtype=np.uint32)
    if values.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers, got dtype {values.dtype}")
    return (values.astype(np.int64, copy=False) & np.int64(0xFFFFFFFF)).astype(np.uint32)


def _normalize_prefiltered_pairs(
    pairs: Sequence[tuple[int, int]] | np.ndarray,
    shape_count: int,
) -> np.ndarray:
    """Return sorted unique canonical prefiltered pairs."""
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"prefiltered_pairs must have shape [pair_count, 2], got {pairs.shape}")
    if np.any(pairs < 0) or np.any(pairs >= shape_count):
        raise ValueError(f"prefiltered_pairs contains shape indices outside [0, {shape_count})")
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)
    codes = (pairs[:, 0] << 32) | pairs[:, 1]
    order = np.argsort(codes)
    pairs = pairs[order]
    codes = codes[order]
    pairs = pairs[np.concatenate(([True], codes[1:] != codes[:-1]))]
    return pairs.astype(np.int32, copy=False)


def _build_class_problem(
    collision_type: np.ndarray,
    collision_affinity: np.ndarray,
    prefiltered_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compress shape masks and pair multiplicities into a weighted class graph."""
    signatures = np.column_stack((collision_type, collision_affinity))
    classes, inverse, counts = np.unique(signatures, axis=0, return_inverse=True, return_counts=True)
    class_count = classes.shape[0]

    candidate_count = counts[:, None].astype(np.int64) * counts[None, :].astype(np.int64)
    candidate_count[np.diag_indices(class_count)] = counts.astype(np.int64) * (counts.astype(np.int64) - 1) // 2

    if prefiltered_pairs.shape[0]:
        pair_classes = inverse[prefiltered_pairs]
        lo = np.minimum(pair_classes[:, 0], pair_classes[:, 1])
        hi = np.maximum(pair_classes[:, 0], pair_classes[:, 1])
        removed = np.bincount(lo * class_count + hi, minlength=class_count * class_count).reshape(
            (class_count, class_count)
        )
        candidate_count -= removed
        candidate_count -= np.triu(removed, 1).T

    class_type = classes[:, 0]
    class_affinity = classes[:, 1]
    allowed = ((class_type[:, None] & class_affinity[None, :]) != 0) | (
        (class_type[None, :] & class_affinity[:, None]) != 0
    )
    return classes, inverse, candidate_count, allowed


def _assignment_cost(
    assignment: np.ndarray,
    candidate_count: np.ndarray,
    allowed: np.ndarray,
) -> int | None:
    """Return false-positive pair count, or ``None`` for a false-negative assignment."""
    cost = 0
    for class_a in range(assignment.shape[0]):
        group_a = int(assignment[class_a])
        for class_b in range(class_a, assignment.shape[0]):
            pair_count = int(candidate_count[class_a, class_b])
            if pair_count == 0:
                continue
            group_allowed = _group_pair_allowed(group_a, int(assignment[class_b]))
            if allowed[class_a, class_b]:
                if not group_allowed:
                    return None
            elif group_allowed:
                cost += pair_count
    return cost


def _greedy_hybrid_assignment(
    candidate_count: np.ndarray,
    allowed: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Build a fast feasible assignment using positive classes and DSATUR."""
    class_count = allowed.shape[0]
    required = allowed & (candidate_count > 0)
    assignment = np.zeros(class_count, dtype=np.int32)

    # A class with required within-class contacts cannot share one negative
    # group. Keeping all such classes in one positive group is always feasible.
    positive = active & np.diag(required)
    assignment[positive] = 1

    negative = np.flatnonzero(active & ~positive)
    if negative.shape[0] == 0:
        return assignment

    neighbors = required[np.ix_(negative, negative)].copy()
    np.fill_diagonal(neighbors, False)
    weighted_degree = (neighbors * candidate_count[np.ix_(negative, negative)]).sum(axis=1)
    colors = np.full(negative.shape[0], -1, dtype=np.int32)

    for _ in range(negative.shape[0]):
        uncolored = np.flatnonzero(colors < 0)
        saturation = np.fromiter(
            (np.unique(colors[neighbors[index] & (colors >= 0)]).shape[0] for index in uncolored),
            dtype=np.int32,
            count=uncolored.shape[0],
        )
        selected = int(
            uncolored[
                max(
                    range(uncolored.shape[0]),
                    key=lambda index: (
                        int(saturation[index]),
                        int(weighted_degree[uncolored[index]]),
                        -int(negative[uncolored[index]]),
                    ),
                )
            ]
        )
        forbidden = set(colors[neighbors[selected] & (colors >= 0)].tolist())
        color = 0
        while color in forbidden:
            color += 1
        colors[selected] = color

    assignment[negative] = -1 - colors
    return assignment


def _search_class_assignment(
    candidate_count: np.ndarray,
    allowed: np.ndarray,
    force_active: np.ndarray,
    max_search_nodes: int,
    max_exact_classes: int,
    max_positive_groups: int | None,
) -> tuple[np.ndarray, int, int, bool]:
    """Find the minimum-exclusion class-homogeneous group assignment."""
    class_count = allowed.shape[0]
    required = allowed & (candidate_count > 0)
    active = required.any(axis=0) | required.any(axis=1) | force_active

    baseline = np.where(active, 1, 0).astype(np.int32)
    best_assignment = baseline.copy()
    best_cost = _assignment_cost(best_assignment, candidate_count, allowed)
    if best_cost is None:
        raise RuntimeError("The all-positive baseline must always be feasible")
    best_key = (best_cost, int(np.unique(best_assignment[best_assignment != 0]).shape[0]))

    greedy = _greedy_hybrid_assignment(candidate_count, allowed, active)
    greedy_cost = _assignment_cost(greedy, candidate_count, allowed)
    if greedy_cost is None:
        raise RuntimeError("The greedy hybrid assignment must always be feasible")
    greedy_key = (greedy_cost, int(np.unique(greedy[greedy != 0]).shape[0]))
    if greedy_key < best_key:
        best_assignment = greedy
        best_cost = greedy_cost
        best_key = greedy_key

    if class_count == 0 or not np.any(active):
        return best_assignment, 1, 0, True
    if int(np.count_nonzero(active)) > max_exact_classes:
        return best_assignment, 0, best_cost, False

    weighted_degree = (candidate_count * required).sum(axis=0) + (candidate_count * required).sum(axis=1)
    order = np.flatnonzero(active)
    order = order[np.argsort(-weighted_degree[order], kind="stable")]

    assignment = np.zeros(class_count, dtype=np.int32)
    search_nodes = 0
    aborted = False

    def visit(depth: int, positive_count: int, negative_count: int, cost: int) -> None:
        nonlocal aborted, best_assignment, best_cost, best_key, search_nodes
        if aborted:
            return
        search_nodes += 1
        if search_nodes > max_search_nodes:
            aborted = True
            return
        if cost > best_cost:
            return
        if depth == order.shape[0]:
            used_groups = int(np.unique(assignment[assignment != 0]).shape[0])
            key = (cost, used_groups)
            if key < best_key:
                best_assignment = assignment.copy()
                best_cost = cost
                best_key = key
            return

        class_index = int(order[depth])
        labels = [*range(1, positive_count + 1)]
        if max_positive_groups is None or positive_count < max_positive_groups:
            labels.append(positive_count + 1)
        if not required[class_index, class_index]:
            labels.extend([*range(-1, -negative_count - 1, -1), -(negative_count + 1)])

        candidates: list[tuple[int, bool, int]] = []
        for group in labels:
            incremental_cost = 0
            group_allowed = _group_pair_allowed(group, group)
            self_count = int(candidate_count[class_index, class_index])
            if self_count:
                if allowed[class_index, class_index]:
                    if not group_allowed:
                        continue
                elif group_allowed:
                    incremental_cost += self_count

            feasible = True
            for previous_depth in range(depth):
                other = int(order[previous_depth])
                pair_count = int(candidate_count[min(class_index, other), max(class_index, other)])
                if pair_count == 0:
                    continue
                pair_allowed = _group_pair_allowed(group, int(assignment[other]))
                if allowed[class_index, other]:
                    if not pair_allowed:
                        feasible = False
                        break
                elif pair_allowed:
                    incremental_cost += pair_count
            if feasible and cost + incremental_cost <= best_cost:
                is_new = (group > 0 and group > positive_count) or (group < 0 and -group > negative_count)
                candidates.append((incremental_cost, is_new, group))

        candidates.sort()
        for incremental_cost, _is_new, group in candidates:
            assignment[class_index] = group
            visit(
                depth + 1,
                max(positive_count, group if group > 0 else 0),
                max(negative_count, -group if group < 0 else 0),
                cost + incremental_cost,
            )
            assignment[class_index] = 0
            if aborted:
                return

    visit(0, 0, 0, 0)
    return best_assignment, search_nodes, best_cost, not aborted


def _pack_pair_codes(pairs: np.ndarray) -> np.ndarray:
    """Pack canonical int32 shape pairs into sorted uint64 codes."""
    if pairs.shape[0] == 0:
        return np.empty(0, dtype=np.uint64)
    return (pairs[:, 0].astype(np.uint64) << np.uint64(32)) | pairs[:, 1].astype(np.uint64)


def _build_excluded_pairs(
    inverse: np.ndarray,
    class_groups: np.ndarray,
    candidate_count: np.ndarray,
    allowed: np.ndarray,
    prefiltered_pairs: np.ndarray,
) -> np.ndarray:
    """Materialize over-admitted pairs directly from incompatible class blocks."""
    prefiltered_codes = _pack_pair_codes(prefiltered_pairs)
    chunks: list[np.ndarray] = []
    class_shapes = [
        np.flatnonzero(inverse == class_index).astype(np.int32) for class_index in range(class_groups.shape[0])
    ]

    for class_a in range(class_groups.shape[0]):
        group_a = int(class_groups[class_a])
        for class_b in range(class_a, class_groups.shape[0]):
            if (
                candidate_count[class_a, class_b] == 0
                or allowed[class_a, class_b]
                or not _group_pair_allowed(group_a, int(class_groups[class_b]))
            ):
                continue

            shapes_a = class_shapes[class_a]
            shapes_b = class_shapes[class_b]
            if class_a == class_b:
                row, column = np.triu_indices(shapes_a.shape[0], 1)
                pairs = np.column_stack((shapes_a[row], shapes_a[column]))
            else:
                pairs = np.column_stack(
                    (
                        np.repeat(shapes_a, shapes_b.shape[0]),
                        np.tile(shapes_b, shapes_a.shape[0]),
                    )
                )
                pairs.sort(axis=1)

            if prefiltered_codes.shape[0] and pairs.shape[0]:
                codes = _pack_pair_codes(pairs)
                locations = np.searchsorted(prefiltered_codes, codes)
                in_range = locations < prefiltered_codes.shape[0]
                is_prefiltered = np.zeros(pairs.shape[0], dtype=bool)
                is_prefiltered[in_range] = prefiltered_codes[locations[in_range]] == codes[in_range]
                pairs = pairs[~is_prefiltered]
            if pairs.shape[0]:
                chunks.append(pairs)

    if not chunks:
        return np.empty((0, 2), dtype=np.int32)
    pairs = np.concatenate(chunks, axis=0).astype(np.int32, copy=False)
    if pairs.shape[0] > 1:
        order = np.argsort(_pack_pair_codes(pairs))
        pairs = pairs[order]
    return pairs


def _validate_prefiltered_pair_bounds(
    pairs: Sequence[tuple[int, int]] | np.ndarray,
    shape_count: int,
) -> None:
    """Validate pair shape and bounds without canonicalizing the collection."""
    pairs = np.asarray(pairs)
    if pairs.size == 0:
        return
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"prefiltered_pairs must have shape [pair_count, 2], got {pairs.shape}")
    if pairs.dtype.kind not in "iu":
        raise TypeError(f"prefiltered_pairs must contain integers, got dtype {pairs.dtype}")
    if np.any(pairs < 0) or np.any(pairs >= shape_count):
        raise ValueError(f"prefiltered_pairs contains shape indices outside [0, {shape_count})")


def compile_collision_masks(
    collision_type: Sequence[int] | np.ndarray,
    collision_affinity: Sequence[int] | np.ndarray,
    *,
    prefiltered_pairs: Sequence[tuple[int, int]] | np.ndarray = (),
    force_nonzero: bool = False,
    max_search_nodes: int = 50_000,
    max_exact_classes: int = 8,
    max_positive_groups: int | None = None,
) -> CollisionMaskCompileResult:
    """Compile MuJoCo masks into exact Newton groups and exclusion pairs.

    The compiler compresses identical mask signatures into weighted graph
    vertices. A branch-and-bound search assigns positive and negative Newton
    groups while enforcing that every MuJoCo-compatible pair remains enabled.
    Any incompatible pair still enabled by the selected groups is emitted as an
    exclusion pair, so the resulting relation is exact even when the bounded
    optimizer falls back to the all-positive baseline.

    Existing same-body, parent-child, or explicitly excluded pairs may be
    supplied through ``prefiltered_pairs``. They are removed from both the
    optimizer's cost and the generated exclusions.

    Args:
        collision_type: MuJoCo ``contype`` value per shape.
        collision_affinity: MuJoCo ``conaffinity`` value per shape.
        prefiltered_pairs: Pairs already excluded by another Newton mechanism.
        force_nonzero: Keep every nonzero mask class in a nonzero Newton group
            even when all of its local pairs are already prefiltered. This
            preserves collision eligibility with shapes outside the compiled
            collection.
        max_search_nodes: Maximum branch-and-bound states. The best exact
            lowering found so far is returned if this budget is exhausted.
        max_exact_classes: Maximum number of active mask classes optimized by
            branch-and-bound. Larger problems use the exact all-positive and
            greedy hybrid candidates without an exponential search.
        max_positive_groups: Optional limit on positive Newton groups. Importers
            can set this to one so every imported shape remains compatible with
            the builder's existing default positive group.

    Returns:
        The normalized collision groups and additional exclusion pairs.
    """
    if max_search_nodes <= 0:
        raise ValueError(f"max_search_nodes must be positive, got {max_search_nodes}")
    if max_exact_classes <= 0:
        raise ValueError(f"max_exact_classes must be positive, got {max_exact_classes}")
    if max_positive_groups is not None and max_positive_groups <= 0:
        raise ValueError(f"max_positive_groups must be positive or None, got {max_positive_groups}")

    collision_type = _normalize_masks(collision_type, "collision_type")
    collision_affinity = _normalize_masks(collision_affinity, "collision_affinity")
    if collision_type.shape != collision_affinity.shape:
        raise ValueError(
            "collision_type and collision_affinity must have the same shape, "
            f"got {collision_type.shape} and {collision_affinity.shape}"
        )
    shape_count = collision_type.shape[0]
    canonical_disabled = (collision_type == 0) & (collision_affinity == 0)
    canonical_default = (collision_type == 1) & (collision_affinity == 1)
    if np.all(canonical_disabled | canonical_default):
        _validate_prefiltered_pair_bounds(prefiltered_pairs, shape_count)
        groups = canonical_default.astype(np.int32)
        class_count = int(np.any(canonical_disabled)) + int(np.any(canonical_default))
        return CollisionMaskCompileResult(
            groups=groups,
            excluded_pairs=np.empty((0, 2), dtype=np.int32),
            class_count=class_count,
            group_count=int(np.any(canonical_default)),
            search_nodes=0,
            optimal=True,
        )

    prefiltered_pairs = _normalize_prefiltered_pairs(prefiltered_pairs, shape_count)
    classes, inverse, candidate_count, allowed = _build_class_problem(
        collision_type,
        collision_affinity,
        prefiltered_pairs,
    )
    force_active = (classes[:, 0] | classes[:, 1]) != 0 if force_nonzero else np.zeros(classes.shape[0], dtype=bool)
    class_groups, search_nodes, cost, optimal = _search_class_assignment(
        candidate_count,
        allowed,
        force_active,
        max_search_nodes,
        max_exact_classes,
        max_positive_groups,
    )
    groups = class_groups[inverse].astype(np.int32, copy=False)
    excluded_pairs = (
        np.empty((0, 2), dtype=np.int32)
        if cost == 0
        else _build_excluded_pairs(inverse, class_groups, candidate_count, allowed, prefiltered_pairs)
    )
    if excluded_pairs.shape[0] != cost:
        raise RuntimeError(f"Expected {cost} compiled exclusion pairs, materialized {excluded_pairs.shape[0]}")
    group_count = int(np.unique(groups[groups != 0]).shape[0])
    return CollisionMaskCompileResult(
        groups=groups,
        excluded_pairs=excluded_pairs,
        class_count=int(classes.shape[0]),
        group_count=group_count,
        search_nodes=search_nodes,
        optimal=optimal,
    )


def verify_collision_mask_compilation(
    collision_type: Sequence[int] | np.ndarray,
    collision_affinity: Sequence[int] | np.ndarray,
    result: CollisionMaskCompileResult,
    *,
    prefiltered_pairs: Sequence[tuple[int, int]] | np.ndarray = (),
) -> None:
    """Raise if a compiled result differs from the MuJoCo mask predicate."""
    collision_type = _normalize_masks(collision_type, "collision_type")
    collision_affinity = _normalize_masks(collision_affinity, "collision_affinity")
    shape_count = collision_type.shape[0]
    if collision_affinity.shape != collision_type.shape:
        raise ValueError("collision_type and collision_affinity must have the same shape")
    if result.groups.shape != (shape_count,):
        raise ValueError(f"result.groups must have shape {(shape_count,)}, got {result.groups.shape}")

    prefiltered = set(map(tuple, _normalize_prefiltered_pairs(prefiltered_pairs, shape_count).tolist()))
    excluded = set(map(tuple, _normalize_prefiltered_pairs(result.excluded_pairs, shape_count).tolist()))
    for shape_a in range(shape_count - 1):
        for shape_b in range(shape_a + 1, shape_count):
            pair = (shape_a, shape_b)
            if pair in prefiltered:
                continue
            expected = bool(
                (collision_type[shape_a] & collision_affinity[shape_b])
                or (collision_type[shape_b] & collision_affinity[shape_a])
            )
            actual = _group_pair_allowed(int(result.groups[shape_a]), int(result.groups[shape_b]))
            actual = actual and pair not in excluded
            if actual != expected:
                raise AssertionError(
                    f"Pair {pair} differs: masks expect {expected}, groups/exclusions produce {actual}"
                )


def _normalize_collision_groups(values: Sequence[int] | np.ndarray) -> np.ndarray:
    """Normalize Newton collision groups to a one-dimensional int64 array."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"collision_groups must be one-dimensional, got shape {values.shape}")
    if values.dtype.kind not in "iu":
        raise TypeError(f"collision_groups must contain integers, got dtype {values.dtype}")
    return values.astype(np.int64, copy=False)


def _newton_collision_matrix(
    collision_groups: np.ndarray,
    excluded_pairs: np.ndarray,
) -> np.ndarray:
    """Build the exact Newton collision relation for distinct shape pairs."""
    group_a = collision_groups[:, None]
    group_b = collision_groups[None, :]
    allowed = (group_a != 0) & (group_b != 0)
    allowed &= ((group_a > 0) & ((group_a == group_b) | (group_b < 0))) | ((group_a < 0) & (group_a != group_b))
    np.fill_diagonal(allowed, False)
    if excluded_pairs.shape[0]:
        allowed[excluded_pairs[:, 0], excluded_pairs[:, 1]] = False
        allowed[excluded_pairs[:, 1], excluded_pairs[:, 0]] = False
    return allowed


def _candidate_biclique_edges(left: np.ndarray, right: np.ndarray, shape_count: int) -> np.ndarray:
    """Return flattened upper-triangle edges contributed by one mask bit."""
    edge_matrix = (left[:, None] & right[None, :]) | (right[:, None] & left[None, :])
    row, column = np.nonzero(np.triu(edge_matrix, 1))
    return row.astype(np.int64) * shape_count + column


def _disjoint_biclique_edges(left: np.ndarray, right: np.ndarray, shape_count: int) -> np.ndarray:
    """Return flattened edges for disjoint endpoint index arrays."""
    row = np.repeat(left, right.shape[0])
    column = np.tile(right, left.shape[0])
    edge_lo = np.minimum(row, column)
    edge_hi = np.maximum(row, column)
    return edge_lo.astype(np.int64) * shape_count + edge_hi


def compile_newton_collision_graph(
    collision_groups: Sequence[int] | np.ndarray,
    *,
    excluded_pairs: Sequence[tuple[int, int]] | np.ndarray = (),
    max_bits: int = 32,
    max_shape_count: int | None = NEWTON_COLLISION_MASK_MAX_SHAPE_COUNT,
    max_excluded_pair_count: int | None = NEWTON_COLLISION_MASK_MAX_EXCLUDED_PAIR_COUNT,
) -> CollisionGraphCompileResult:
    """Compile Newton groups and exclusions into exact MuJoCo masks when possible.

    Each MuJoCo mask bit describes a complete bipartite subgraph: shapes with
    that bit in ``contype`` collide with shapes carrying it in
    ``conaffinity``. The reverse conversion is therefore a biclique-cover
    problem. This compiler greedily covers the exact Newton pair graph with
    safe group-derived bicliques and per-shape stars. It never emits a false
    positive. Results with ``exact=False`` must not be used because at least
    one required Newton pair remains uncovered.

    The per-shape stars guarantee an exact result for at most 33 shapes. Larger
    structured group graphs commonly need far fewer bits, but arbitrary
    Newton pair filters may exceed MuJoCo's 32-bit capacity. Finding the
    minimum biclique cover is NP-hard, so failure for a larger graph does not
    prove that no alternative 32-bit cover exists.

    The greedy cover uses a dense shape-pair matrix and repeatedly scores its
    candidates. Large shape sets and heavily filtered graphs are therefore
    left to the caller's fallback without constructing that matrix.

    Args:
        collision_groups: Newton signed collision group per shape.
        excluded_pairs: Canonical or noncanonical Newton exclusion pairs.
        max_bits: Maximum MuJoCo mask bits available.
        max_shape_count: Largest graph to compile, or ``None`` for no limit.
        max_excluded_pair_count: Largest sparse filter set to compile, or
            ``None`` for no limit.

    Returns:
        The exact masks when they fit, or an inexact diagnostic result.
    """
    if isinstance(max_bits, bool) or not isinstance(max_bits, (int, np.integer)):
        raise TypeError(f"max_bits must be an integer, got {max_bits!r}")
    if max_shape_count is not None and (
        isinstance(max_shape_count, bool) or not isinstance(max_shape_count, (int, np.integer))
    ):
        raise TypeError(f"max_shape_count must be an integer or None, got {max_shape_count!r}")
    if max_excluded_pair_count is not None and (
        isinstance(max_excluded_pair_count, bool) or not isinstance(max_excluded_pair_count, (int, np.integer))
    ):
        raise TypeError(f"max_excluded_pair_count must be an integer or None, got {max_excluded_pair_count!r}")
    if max_bits <= 0 or max_bits > 32:
        raise ValueError(f"max_bits must be in [1, 32], got {max_bits}")
    if max_shape_count is not None and max_shape_count <= 0:
        raise ValueError(f"max_shape_count must be positive or None, got {max_shape_count}")
    if max_excluded_pair_count is not None and max_excluded_pair_count < 0:
        raise ValueError(f"max_excluded_pair_count must be nonnegative or None, got {max_excluded_pair_count}")

    collision_groups = _normalize_collision_groups(collision_groups)
    shape_count = collision_groups.shape[0]
    excluded_pairs = _normalize_prefiltered_pairs(excluded_pairs, shape_count)

    collision_type = np.zeros(shape_count, dtype=np.uint32)
    collision_affinity = np.zeros(shape_count, dtype=np.uint32)
    if (max_shape_count is not None and shape_count > max_shape_count) or (
        max_excluded_pair_count is not None and excluded_pairs.shape[0] > max_excluded_pair_count
    ):
        return CollisionGraphCompileResult(
            collision_type=collision_type,
            collision_affinity=collision_affinity,
            bit_count=0,
            exact=False,
            uncovered_pair_count=None,
            skipped=True,
        )

    allowed = _newton_collision_matrix(collision_groups, excluded_pairs)
    allowed_flat = allowed.reshape(-1)
    uncovered = np.triu(allowed, 1).reshape(-1).copy()
    if not np.any(uncovered):
        return CollisionGraphCompileResult(
            collision_type=collision_type,
            collision_affinity=collision_affinity,
            bit_count=0,
            exact=True,
            uncovered_pair_count=0,
            skipped=False,
        )

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    candidate_keys: set[tuple[bytes, bytes]] = set()

    def add_candidate(
        left: np.ndarray,
        right: np.ndarray,
        edges: np.ndarray | None = None,
    ) -> None:
        left = np.asarray(left, dtype=bool)
        right = np.asarray(right, dtype=bool)
        if not np.any(left) or not np.any(right):
            return
        left_key = np.packbits(left).tobytes()
        right_key = np.packbits(right).tobytes()
        if right_key < left_key:
            left, right = right, left
            left_key, right_key = right_key, left_key
        key = (left_key, right_key)
        if key in candidate_keys:
            return
        if edges is None:
            edges = _candidate_biclique_edges(left, right, shape_count)
        if edges.shape[0] == 0 or not np.all(allowed_flat[edges]):
            return
        candidate_keys.add(key)
        candidates.append((left, right, edges))

    nonzero_groups = np.unique(collision_groups[collision_groups != 0])
    positive_groups = nonzero_groups[nonzero_groups > 0]
    negative_groups = nonzero_groups[nonzero_groups < 0]

    # One clique bit exactly covers an unfiltered positive group.
    for group in positive_groups:
        members = collision_groups == group
        add_candidate(members, members)

    # All positive-negative pairs share one biclique when no sparse filter
    # punches a hole in it.
    add_candidate(collision_groups > 0, collision_groups < 0)

    # Distinct negative groups form a complete multipartite graph. Binary
    # group codes cover it in ceil(log2(group_count)) bits when unfiltered.
    if negative_groups.shape[0] > 1:
        negative_code = {int(group): index for index, group in enumerate(negative_groups)}
        code_bits = (negative_groups.shape[0] - 1).bit_length()
        for bit in range(code_bits):
            left = np.fromiter(
                (group < 0 and not (negative_code[int(group)] & (1 << bit)) for group in collision_groups),
                dtype=bool,
                count=shape_count,
            )
            right = np.fromiter(
                (group < 0 and bool(negative_code[int(group)] & (1 << bit)) for group in collision_groups),
                dtype=bool,
                count=shape_count,
            )
            add_candidate(left, right)

    # Whole group-pair bicliques retain useful coverage when other group
    # relations contain sparse exclusions.
    for index, group_a in enumerate(nonzero_groups):
        members_a = collision_groups == group_a
        members_a_indices = np.flatnonzero(members_a)
        for group_b in nonzero_groups[index + 1 :]:
            if _group_pair_allowed(int(group_a), int(group_b)):
                members_b = collision_groups == group_b
                edges = _disjoint_biclique_edges(
                    members_a_indices,
                    np.flatnonzero(members_b),
                    shape_count,
                )
                add_candidate(members_a, members_b, edges)

    # A star is always a safe exact biclique and guarantees at most n - 1
    # bits for any n-vertex graph.
    for shape in range(shape_count):
        left = np.zeros(shape_count, dtype=bool)
        left[shape] = True
        neighbors = np.flatnonzero(allowed[shape])
        edge_lo = np.minimum(shape, neighbors)
        edge_hi = np.maximum(shape, neighbors)
        edges = edge_lo.astype(np.int64) * shape_count + edge_hi
        add_candidate(left, allowed[shape], edges)

    selected: list[tuple[np.ndarray, np.ndarray]] = []
    while np.any(uncovered) and len(selected) < max_bits:
        best_candidate = None
        best_score = 0
        for left, right, edges in candidates:
            score = int(np.count_nonzero(uncovered[edges]))
            if score > best_score:
                best_candidate = (left, right, edges)
                best_score = score
        if best_candidate is None:
            break
        left, right, edges = best_candidate
        selected.append((left, right))
        uncovered[edges] = False

    uncovered_pair_count = int(np.count_nonzero(uncovered))
    exact = uncovered_pair_count == 0
    for bit_index, (left, right) in enumerate(selected):
        bit = np.uint32(1) << np.uint32(bit_index)
        collision_type[left] |= bit
        collision_affinity[right] |= bit

    return CollisionGraphCompileResult(
        collision_type=collision_type,
        collision_affinity=collision_affinity,
        bit_count=len(selected) if exact else min(max_bits + 1, len(selected) + 1),
        exact=exact,
        uncovered_pair_count=uncovered_pair_count,
        skipped=False,
    )


def verify_newton_collision_graph_compilation(
    collision_groups: Sequence[int] | np.ndarray,
    result: CollisionGraphCompileResult,
    *,
    excluded_pairs: Sequence[tuple[int, int]] | np.ndarray = (),
) -> None:
    """Raise if MuJoCo masks differ from Newton groups and exclusions."""
    collision_groups = _normalize_collision_groups(collision_groups)
    shape_count = collision_groups.shape[0]
    if result.collision_type.shape != (shape_count,) or result.collision_affinity.shape != (shape_count,):
        raise ValueError("compiled collision masks must contain one value per collision group")
    excluded_pairs = _normalize_prefiltered_pairs(excluded_pairs, shape_count)
    expected = _newton_collision_matrix(collision_groups, excluded_pairs)
    actual = ((result.collision_type[:, None] & result.collision_affinity[None, :]) != 0) | (
        (result.collision_type[None, :] & result.collision_affinity[:, None]) != 0
    )
    np.fill_diagonal(actual, False)
    if not np.array_equal(actual, expected):
        mismatch = np.argwhere(np.triu(actual != expected, 1))
        shape_a, shape_b = map(int, mismatch[0])
        raise AssertionError(
            f"Pair {(shape_a, shape_b)} differs: Newton expects "
            f"{bool(expected[shape_a, shape_b])}, masks produce {bool(actual[shape_a, shape_b])}"
        )


__all__ = [
    "MUJOCO_COLLISION_MASK_DOMAIN_UNSET",
    "MUJOCO_COLLISION_MASK_UNSET",
    "NEWTON_COLLISION_MASK_MAX_EXCLUDED_PAIR_COUNT",
    "NEWTON_COLLISION_MASK_MAX_SHAPE_COUNT",
    "CollisionGraphCompileResult",
    "CollisionMaskCompileResult",
    "compile_collision_masks",
    "compile_newton_collision_graph",
    "mujoco_mask_to_signed",
    "verify_collision_mask_compilation",
    "verify_newton_collision_graph_compilation",
]
