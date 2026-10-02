# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact collision topology and on-demand explicit shape pairs."""

from collections import Counter, defaultdict
from functools import cached_property

import numpy as np

from ..geometry.flags import ShapeFlags

# Bound enumeration and replicated replay scratch independently of scene size.
_PAIR_CHUNK_SIZE = 262144


def _group_counts_by_world(groups_by_world: dict[int, Counter]) -> dict[int, int]:
    """Count compatible groups within worlds and against global shapes."""

    def within(groups):
        positive = negative = pairs = 0
        for group, count in groups.items():
            same_group_pairs = count * (count - 1) // 2
            if group > 0:
                positive += count
                pairs += same_group_pairs
            else:
                negative += count
                pairs -= same_group_pairs
        return pairs + positive * negative + negative * (negative - 1) // 2

    globals_ = groups_by_world.get(-1, {})
    global_total = sum(globals_.values())
    global_negative = sum(count for group, count in globals_.items() if group < 0)
    result = {-1: within(globals_)}
    for world, groups in groups_by_world.items():
        if world < 0:
            continue
        cross = sum(
            count * (global_negative + globals_.get(group, 0) if group > 0 else global_total - globals_.get(group, 0))
            for group, count in groups.items()
        )
        result[world] = within(groups) + cross
    return result


class _ShapeContactPairs:
    """Own a builder-independent snapshot until explicit pairs are requested."""

    def __init__(self, shape_body, shape_world, shape_group, shape_flags, filter_pairs, world_count):
        self.shape_body = np.maximum(np.asarray(shape_body, dtype=np.int32), -1)
        self.shape_world = np.array(shape_world, dtype=np.int32, copy=True)
        self.shape_group = np.array(shape_group, dtype=np.int32, copy=True)
        self.shape_flags = np.array(shape_flags, dtype=np.int32, copy=True)
        self.filter_pairs = np.asarray(filter_pairs, dtype=np.int64)
        self.world_count = world_count
        for array in (self.shape_body, self.shape_world, self.shape_group, self.shape_flags, self.filter_pairs):
            array.setflags(write=False)

    def _active(self, shape_mask: np.ndarray | None = None) -> np.ndarray:
        active = ((self.shape_flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0) & (self.shape_group != 0)
        if shape_mask is not None:
            active &= shape_mask
        return active

    def _active_filters(self, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = (self.filter_pairs >> 32).astype(np.int32)
        second = (self.filter_pairs & 0xFFFFFFFF).astype(np.int32)
        world_a, world_b = self.shape_world[first], self.shape_world[second]
        group_a, group_b = self.shape_group[first], self.shape_group[second]
        keep = (
            active[first]
            & active[second]
            & (self.shape_body[first] != self.shape_body[second])
            & ((world_a == world_b) | (world_a == -1) | (world_b == -1))
            & np.where(group_a > 0, (group_a == group_b) | (group_b < 0), group_a != group_b)
        )
        return first[keep], second[keep]

    @cached_property
    def counts(self) -> np.ndarray:
        """Exact pair counts: globals in slot zero, local world w in slot w + 1."""
        return self.count_pairs()

    def count_pairs(self, shape_mask: np.ndarray | None = None) -> np.ndarray:
        """Count from group and body cardinalities without enumerating pairs."""
        active = self._active(shape_mask)
        groups_array = self.shape_group[active]
        if len(groups_array) > 256 and groups_array[0] > 0 and np.all(groups_array == groups_array[0]):
            # The common unpartitioned collision group needs only per-world and
            # per-body cardinalities. Keep large replicated scenes in NumPy.
            slots = self.shape_world[active].astype(np.int64) + 1
            sizes = np.bincount(slots, minlength=self.world_count + 1).astype(np.int64)
            counts = sizes * (sizes - 1) // 2
            counts[1:] += sizes[0] * sizes[1:]
            codes = (slots << 32) | (self.shape_body[active].astype(np.int64) + 1)
            codes, sizes = np.unique(codes, return_counts=True)
            slots = codes >> 32
            np.subtract.at(counts, slots, sizes * (sizes - 1) // 2)
            global_count = int(np.count_nonzero(slots == 0))
            if global_count:
                body_codes = codes & 0xFFFFFFFF
                positions = np.searchsorted(body_codes[:global_count], body_codes[global_count:])
                clipped = np.minimum(positions, global_count - 1)
                matches = (positions < global_count) & (body_codes[clipped] == body_codes[global_count:])
                np.subtract.at(
                    counts,
                    slots[global_count:][matches],
                    sizes[global_count:][matches] * sizes[clipped[matches]],
                )
            first, second = self._active_filters(active)
            counts -= np.bincount(
                np.maximum(self.shape_world[first], self.shape_world[second]) + 1, minlength=len(counts)
            )
            return counts

        bodies = self.shape_body[active].tolist()
        worlds = self.shape_world[active].tolist()
        groups = self.shape_group[active].tolist()
        body_counts = Counter(bodies)
        world_groups = defaultdict(Counter)
        body_world_groups = defaultdict(lambda: defaultdict(Counter))
        for body, world, group in zip(bodies, worlds, groups, strict=True):
            world_groups[world][group] += 1
            if body_counts[body] > 1:
                body_world_groups[body][world][group] += 1

        counts = np.zeros(self.world_count + 1, dtype=np.int64)
        for world, count in _group_counts_by_world(world_groups).items():
            counts[world + 1] = count
        # Use sparse world maps per body: a dense world-by-body intermediate
        # would reintroduce quadratic memory for replicated scenes.
        for body_groups in body_world_groups.values():
            for world, count in _group_counts_by_world(body_groups).items():
                counts[world + 1] -= count

        first, second = self._active_filters(active)
        pair_world = np.maximum(self.shape_world[first], self.shape_world[second])
        counts -= np.bincount(pair_world + 1, minlength=len(counts))
        return counts

    def _store_pairs(self, output, a, b) -> int:
        """Canonicalize candidate indices and remove authored exclusions."""
        shape_a, shape_b = np.minimum(a, b), np.maximum(a, b)
        if len(self.filter_pairs) and len(shape_a):
            codes = (shape_a.astype(np.int64) << 32) | shape_b
            positions = np.searchsorted(self.filter_pairs, codes)
            excluded = (positions < len(self.filter_pairs)) & (
                self.filter_pairs[np.minimum(positions, len(self.filter_pairs) - 1)] == codes
            )
            shape_a, shape_b = shape_a[~excluded], shape_b[~excluded]
        output[: len(shape_a), 0] = shape_a
        output[: len(shape_a), 1] = shape_b
        return len(shape_a)

    def _write_dense_pairs(self, output, first, second, *, triangular) -> int:
        """Fill compatible uniform groups with cache-sized rectangular blocks."""
        written = 0
        rows_per_chunk = max(1, _PAIR_CHUNK_SIZE // len(second))
        for start in range(0, len(first), rows_per_chunk):
            rows = first[start : start + rows_per_chunk]
            columns = second[start + 1 :] if triangular else second
            for column_start in range(0, len(columns), _PAIR_CHUNK_SIZE):
                cols = columns[column_start : column_start + _PAIR_CHUNK_SIZE]
                keep = self.shape_body[rows, None] != self.shape_body[cols]
                if triangular:
                    keep &= rows[:, None] < cols
                row, column = np.nonzero(keep)
                written += self._store_pairs(output[written:], rows[row], cols[column])
        return written

    def _write_pair_ranges(self, output, first, second, starts, ends) -> int:
        """Emit ragged rows of group-compatible candidates in bounded chunks."""
        sizes = ends - starts
        offsets = np.empty(len(first) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(sizes, out=offsets[1:])
        written = 0
        for begin in range(0, int(offsets[-1]), _PAIR_CHUNK_SIZE):
            end = min(begin + _PAIR_CHUNK_SIZE, int(offsets[-1]))
            first_row = int(np.searchsorted(offsets, begin, side="right")) - 1
            last_row = int(np.searchsorted(offsets, end - 1, side="right"))
            repeats = sizes[first_row:last_row].copy()
            repeats[0] -= begin - offsets[first_row]
            repeats[-1] -= offsets[last_row] - end
            rows = np.repeat(np.arange(first_row, last_row), repeats)
            columns = starts[rows] + np.arange(begin, end) - offsets[rows]
            a, b = first[rows], second[columns]
            keep = self.shape_body[a] != self.shape_body[b]
            written += self._store_pairs(output[written:], a[keep], b[keep])
        return written

    def _write_pairs(
        self,
        output: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        *,
        triangular: bool = False,
    ) -> int:
        """Enumerate compatible groups while skipping dominant same-body blocks."""
        if not len(first) or not len(second):
            return 0

        # A body occupying most of both sets can otherwise cause quadratic work
        # for only a linear number of output pairs. Split away that diagonal.
        # Small world templates need no partitioning overhead.
        if len(first) * len(second) > _PAIR_CHUNK_SIZE:
            bodies, counts = np.unique(self.shape_body[first], return_counts=True)
            largest = int(np.argmax(counts))
            if counts[largest] * 2 > len(first):
                same_first = self.shape_body[first] == bodies[largest]
                same_second = same_first if triangular else self.shape_body[second] == bodies[largest]
                if np.count_nonzero(same_second) * 2 > len(second):
                    other_first, other_second = first[~same_first], second[~same_second]
                    count = self._write_pairs(output, other_first, other_second, triangular=triangular)
                    count += self._write_pairs(output[count:], first[same_first], other_second)
                    if not triangular:
                        count += self._write_pairs(output[count:], other_first, second[same_second])
                    return count

        groups = self.shape_group[first]
        second_groups = groups if triangular else self.shape_group[second]
        if np.all(groups == groups[0]) and (triangular or np.all(second_groups == second_groups[0])):
            group_a, group_b = int(groups[0]), int(second_groups[0])
            compatible = (group_a == group_b or group_b < 0) if group_a > 0 else group_a != group_b
            if not compatible:
                return 0
            return self._write_dense_pairs(output, first, second, triangular=triangular)

        # Sorting by (group, body) makes both a positive group and its excluded
        # same-body runs contiguous, even when no body dominates the whole set.
        codes = (second_groups.astype(np.int64) << 32) | (self.shape_body[second].astype(np.int64) + 1)
        order = np.argsort(codes, kind="stable")
        codes = codes[order]
        second = second[order]
        second_groups = second_groups[order]
        if triangular:
            # Negative groups collide with every later group; positive groups
            # collide only within themselves. Each row is one contiguous range.
            first = second
            group_ends = np.searchsorted(second_groups, second_groups, side="right")
            body_ends = np.searchsorted(codes, codes, side="right")
            starts = np.where(second_groups < 0, group_ends, body_ends)
            ends = np.where(second_groups < 0, len(second), group_ends)
            return self._write_pair_ranges(output, first, second, starts, ends)

        # Cross-world/global pairs need at most three ranges per row: negative
        # groups accept everything except their own group; positives accept all
        # negatives and their matching positive group, excluding the same body.
        left = np.searchsorted(second_groups, groups, side="left")
        right = np.searchsorted(second_groups, groups, side="right")
        first_codes = (groups.astype(np.int64) << 32) | (self.shape_body[first].astype(np.int64) + 1)
        body_left = np.searchsorted(codes, first_codes, side="left")
        body_right = np.searchsorted(codes, first_codes, side="right")
        negative_end = int(np.searchsorted(second_groups, 0))
        count = self._write_pair_ranges(
            output, first, second, np.zeros(len(first), dtype=np.int64), np.where(groups < 0, left, negative_end)
        )
        count += self._write_pair_ranges(
            output[count:],
            first,
            second,
            np.where(groups < 0, right, left),
            np.where(groups < 0, len(second), body_left),
        )
        count += self._write_pair_ranges(output[count:], first, second, np.where(groups < 0, right, body_right), right)
        return count

    def build_pairs(self, shape_mask: np.ndarray | None = None) -> np.ndarray:
        """Build only the requested pairs, reusing homogeneous world templates."""
        counts = self.counts if shape_mask is None else self.count_pairs(shape_mask)
        pairs = np.empty((int(counts.sum()), 2), dtype=np.int32)
        if not len(pairs):
            return pairs
        active = self._active(shape_mask)
        indices = np.flatnonzero(active).astype(np.int32)
        indices = indices[np.argsort(self.shape_world[indices], kind="stable")]
        worlds, offsets, sizes = np.unique(self.shape_world[indices], return_index=True, return_counts=True)
        indices_by_world = {
            world: indices[offset : offset + size]
            for world, offset, size in zip(worlds.tolist(), offsets.tolist(), sizes.tolist(), strict=True)
        }
        globals_ = indices_by_world.get(-1, np.empty(0, dtype=np.int32))
        global_bodies = np.unique(self.shape_body[globals_])
        global_bodies = global_bodies[global_bodies >= 0]

        filters_by_world = defaultdict(list)
        first, second = self._active_filters(active)
        for a, b in zip(first.tolist(), second.tolist(), strict=True):
            world = max(int(self.shape_world[a]), int(self.shape_world[b]))
            filters_by_world[world].append((a, b))

        written = self._write_pairs(pairs[: counts[0]], globals_, globals_, triangular=True) if counts[0] else 0
        if written == len(pairs):
            return pairs
        # Normalize all world signatures together. Per-world NumPy operations
        # otherwise dominate setup for thousands of small replicated worlds.
        world_slot = self.shape_world + 1
        starts = np.full(self.world_count + 1, len(self.shape_body), dtype=np.int64)
        indices = np.flatnonzero(active)
        np.minimum.at(starts, world_slot[indices], indices)
        body_starts = np.full(self.world_count + 1, np.iinfo(np.int32).max, dtype=np.int64)
        dynamic = active & (self.shape_body >= 0)
        np.minimum.at(body_starts, world_slot[dynamic], self.shape_body[dynamic])
        layout = np.empty((len(self.shape_body), 4), dtype=np.int64)
        layout[:, 0] = np.arange(len(layout)) - starts[world_slot]
        layout[:, 1] = self.shape_body - body_starts[world_slot]
        layout[self.shape_body < 0, 1] = -1
        attached_to_global = np.isin(self.shape_body, global_bodies)
        layout[attached_to_global, 1] = -2 - self.shape_body[attached_to_global].astype(np.int64)
        layout[:, 2] = self.shape_group
        # Distinguish globals before/between/after local shapes so replay keeps
        # canonical ordering even with interleaved globals and skipped validation.
        layout[:, 3] = np.searchsorted(globals_, np.arange(len(layout)))
        templates = {}
        runs = []
        for world, local in sorted(indices_by_world.items()):
            if world < 0 or not counts[world + 1]:
                continue
            start = int(local[0])
            filter_key = b""
            if world in filters_by_world:
                filters = np.asarray(filters_by_world[world], dtype=np.int32)
                filters = np.where(self.shape_world[filters] == -1, -1 - filters, filters - start)
                filter_key = filters.tobytes()
            key = (layout[local].tobytes(), filter_key)
            count = int(counts[world + 1])
            template = templates.get(key)
            if template is None:
                target = pairs[written : written + count]
                global_count = self._write_pairs(target, globals_, local)
                local_count = self._write_pairs(target[global_count:], local, local, triangular=True)
                if global_count + local_count != count:
                    raise RuntimeError("Shape contact-pair count disagrees with enumeration")
                template = (start, target)
                templates[key] = template
            if runs and runs[-1][0] is template:
                runs[-1][1].append(start)
            else:
                runs.append((template, [start], written))
            written += count

        for (origin, template), run_starts, run_offset in runs:
            count = len(template)
            # Templates are views into the final allocation. Skip writing the
            # original template back onto itself while replaying its run.
            skip = int(run_starts[0] == origin)
            starts = run_starts[skip:]
            offset = run_offset + skip * count
            worlds_per_chunk = max(1, _PAIR_CHUNK_SIZE // count)
            for first_world in range(0, len(starts), worlds_per_chunk):
                shifts = np.asarray(starts[first_world : first_world + worlds_per_chunk], dtype=np.int32) - origin
                end = offset + len(shifts) * count
                target = pairs[offset:end].reshape((len(shifts), count, 2))
                pairs_per_chunk = max(1, _PAIR_CHUNK_SIZE // len(shifts))
                for pair_start in range(0, count, pairs_per_chunk):
                    pair_end = min(pair_start + pairs_per_chunk, count)
                    for column in range(2):
                        values = template[pair_start:pair_end, column]
                        local_mask = self.shape_world[values] != -1
                        target[:, pair_start:pair_end, column] = values + shifts[:, None] * local_mask
                offset = end
        if written != len(pairs):
            raise RuntimeError("Shape contact-pair count disagrees with enumeration")
        return pairs


def _shape_contact_pairs_for_mask(model, shape_mask: np.ndarray, *, shape_pairs=None) -> np.ndarray:
    """Get a subset without materializing the model's full explicit pair cache."""
    data = model._shape_contact_pair_data
    if shape_pairs is None and data is not None:
        return data.build_pairs(shape_mask)
    pairs = model.shape_contact_pairs if shape_pairs is None else shape_pairs
    if pairs is None:
        return np.empty((0, 2), dtype=np.int32)
    pairs = pairs.numpy().reshape((-1, 2))
    return pairs[shape_mask[pairs[:, 0]] & shape_mask[pairs[:, 1]]]


def _shape_contact_pair_count_for_mask(model, shape_mask: np.ndarray, *, shape_pairs=None) -> int:
    """Count a subset without triggering lazy pair construction."""
    data = model._shape_contact_pair_data
    if shape_pairs is None and data is not None:
        return int(data.count_pairs(shape_mask).sum())
    return len(_shape_contact_pairs_for_mask(model, shape_mask, shape_pairs=shape_pairs))
