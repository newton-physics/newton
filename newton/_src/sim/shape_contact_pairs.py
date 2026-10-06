# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collision topology summaries and on-demand explicit shape pairs."""

from collections import defaultdict
from functools import cached_property

import numpy as np

from ..geometry.flags import ShapeFlags

# Bound enumeration and replicated replay scratch independently of scene size.
_PAIR_CHUNK_SIZE = 262144


class _ShapeContactPairs:
    """Finalized collision topology for counting and enumerating collision pairs."""

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

    def count_pairs(
        self,
        shape_mask: np.ndarray | None = None,
        *,
        categories: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Count pairs or sum symmetric category weights without enumerating pairs."""
        active = self._active(shape_mask)
        indices = np.flatnonzero(active)
        counts = np.zeros(self.world_count + 1, dtype=np.int64)
        if not len(indices):
            return counts
        categories = np.zeros(len(self.shape_body), dtype=np.int32) if categories is None else categories
        weights = np.ones((1, 1), dtype=np.int64) if weights is None else weights

        bodies = self.shape_body[indices]
        body_values, body_sizes = np.unique(bodies, return_counts=True)
        repeated = np.isin(bodies, body_values[body_sizes > 1])
        # Owner zero counts all shapes; other owners subtract same-body pairs.
        indices = np.concatenate((indices, indices[repeated]))
        owners = np.concatenate((np.zeros(len(bodies), dtype=np.int64), bodies[repeated].astype(np.int64) + 2))
        slots = self.shape_world[indices].astype(np.int64) + 1
        kinds = categories[indices]
        groups, group_ids = np.unique(self.shape_group[indices], return_inverse=True)
        group_count, kind_count = len(groups), len(weights)
        if not np.any(repeated):
            context_codes, contexts = np.arange(len(counts), dtype=np.int64), slots
        else:
            context_codes, contexts = np.unique(owners * len(counts) + slots, return_inverse=True)
        context_slots = context_codes % len(counts)
        context_owners = context_codes // len(counts)
        # Bound dense group histograms by the number of shapes; sparse groups
        # must not allocate a world-by-group Cartesian product.
        if len(context_codes) * group_count <= 2 * len(indices):
            group_codes = np.arange(len(context_codes) * group_count, dtype=np.int64)
            group_rows = contexts * group_count + group_ids
        else:
            group_codes, group_rows = np.unique(contexts * group_count + group_ids, return_inverse=True)
        group_contexts, group_types = np.divmod(group_codes, group_count)
        kind_codes = contexts * kind_count + kinds
        total = np.bincount(kind_codes, minlength=len(context_codes) * kind_count).reshape(-1, kind_count)
        positive = groups[group_ids] > 0
        pos = np.bincount(kind_codes[positive], minlength=total.size).reshape(total.shape)
        grouped = np.bincount(group_rows * kind_count + kinds, minlength=len(group_codes) * kind_count).reshape(
            -1, kind_count
        )
        signs = np.where(groups[group_types] > 0, 1, -1)

        # Subtract positive-only pairs, then add matching positives and remove
        # matching negatives. Apply the same rules within worlds and to globals.
        rows = np.arange(len(context_codes), dtype=np.int64)
        values = np.zeros(len(context_codes), dtype=np.int64)
        contributions = (
            (total, rows, context_owners, 1),
            (pos, rows, context_owners, -1),
            (grouped, group_contexts, context_owners[group_contexts] * group_count + group_types, signs),
        )
        for hist, rows, keys, sign in contributions:
            coefficients = np.broadcast_to(sign, len(hist))
            weighted = hist @ weights
            within = (np.sum(weighted * hist, axis=1) - hist @ weights.diagonal()) // 2
            np.add.at(values, rows, coefficients * within)
            globals_ = context_slots[rows] == 0
            global_keys = keys[globals_]
            if len(global_keys):
                positions = np.searchsorted(global_keys, keys)
                clipped = np.minimum(positions, len(global_keys) - 1)
                valid = (positions < len(global_keys)) & (global_keys[clipped] == keys) & ~globals_
                cross = np.sum(weighted[valid] * hist[globals_][clipped[valid]], axis=1)
                np.add.at(values, rows[valid], coefficients[valid] * cross)
        np.add.at(counts, context_slots, np.where(context_owners == 0, values, -values))
        first, second = self._active_filters(active)
        np.subtract.at(
            counts,
            np.maximum(self.shape_world[first], self.shape_world[second]) + 1,
            weights[categories[first], categories[second]],
        )
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


def _shape_contact_pair_counts(model, *, shape_mask: np.ndarray | None = None) -> np.ndarray:
    """Cache read-only counts for globals (slot zero) and each local world.

    Default pairs use finalized topology; supplied tables use their active prefix.
    A mask selects pairs whose two shapes are selected.
    """
    if shape_mask is not None:
        shape_mask = np.asarray(shape_mask)
        if shape_mask.dtype != np.bool_ or shape_mask.shape != (model.shape_count,):
            raise ValueError("shape_mask must be a boolean array of length shape_count")
    key = None if shape_mask is None else np.packbits(shape_mask).tobytes()
    supplied, pairs = _shape_contact_pair_override(model)
    use_pairs = supplied or model.shape_contact_pair_count != int(model._shape_contact_pair_data.counts.sum())
    # Supplied arrays and active prefixes can change without invalidating summaries.
    cache = None if use_pairs else model._shape_contact_pair_counts
    if cache is not None and key in cache:
        return cache[key]
    if use_pairs:
        counts = np.zeros(model.world_count + 1, dtype=np.int64)
        if not supplied and model.shape_contact_pair_count:
            pairs = model.shape_contact_pairs
        if pairs is not None:
            pairs = pairs.numpy().reshape((-1, 2))[: model.shape_contact_pair_count]
            if shape_mask is not None:
                pairs = pairs[shape_mask[pairs].all(axis=1)]
            if len(pairs):
                worlds = model.shape_world.numpy() if supplied else model._shape_contact_pair_data.shape_world
                counts = np.bincount(np.max(worlds[pairs], axis=1) + 1, minlength=model.world_count + 1)
    else:
        counts = _shape_contact_pair_topology(model).count_pairs(shape_mask)
    counts.setflags(write=False)
    if cache is not None:
        cache[key] = counts
    return counts


def _shape_contact_pair_topology(model) -> _ShapeContactPairs:
    """Use the finalized Model topology inherited alongside the default pair list."""
    return model._shape_contact_pair_data


def _shape_contact_pair_override(model):
    """Distinguish authored pair lists from a cache of model-derived pairs."""
    overrides = vars(model).get("_overrides")
    if overrides is not None:
        if "shape_contact_pairs" in overrides:
            return True, model.shape_contact_pairs
        supplied, _ = _shape_contact_pair_override(vars(model)["_parent"])
        return supplied, model.shape_contact_pairs if supplied else None
    if model._shape_contact_pair_data is None:
        return True, model.shape_contact_pairs
    return False, None


def _shape_contact_pairs_for_mask(model, shape_mask: np.ndarray, *, shape_pairs=None) -> np.ndarray:
    """Get a subset without materializing the model's full explicit pair cache."""
    supplied, pairs = _shape_contact_pair_override(model)
    if shape_pairs is None and not supplied:
        topology = _shape_contact_pair_topology(model)
        if model.shape_contact_pair_count == int(topology.counts.sum()):
            return topology.build_pairs(shape_mask)
        if model.shape_contact_pair_count == 0:
            return np.empty((0, 2), dtype=np.int32)
        pairs = model.shape_contact_pairs
    pairs = pairs if shape_pairs is None else shape_pairs
    if pairs is None:
        return np.empty((0, 2), dtype=np.int32)
    pairs = pairs.numpy().reshape((-1, 2))
    if shape_pairs is None:
        pairs = pairs[: model.shape_contact_pair_count]
    return pairs[shape_mask[pairs[:, 0]] & shape_mask[pairs[:, 1]]]


def _shape_contact_pair_count_for_mask(model, shape_mask: np.ndarray, *, shape_pairs=None) -> int:
    """Count a subset without triggering lazy pair construction."""
    if shape_pairs is None:
        return int(_shape_contact_pair_counts(model, shape_mask=shape_mask).sum())
    pairs = shape_pairs.numpy().reshape((-1, 2))
    return int(np.count_nonzero(shape_mask[pairs[:, 0]] & shape_mask[pairs[:, 1]]))
