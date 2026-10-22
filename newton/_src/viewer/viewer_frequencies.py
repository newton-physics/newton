# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metadata-backed custom objects, with lazy explorer rows and bounded pages."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..sim.model import Model

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any

    from newton._src.viewer.viewer_model_tree import ModelTree
    from newton._src.viewer.viewer_types import TreeKey


def attribute_value(model: Model, name: str) -> Any:
    """Resolve a registered attribute without assuming a solver namespace."""
    if ":" in name:
        namespace, name = name.split(":", 1)
        model = getattr(model, namespace, None)
    return getattr(model, name, None)


class FrequencyCatalog:
    """Index solver-defined row types so they can appear as explorer objects.

    An attribute frequency describes what each row belongs to, not how often
    it updates. Built-in frequencies include BODY and JOINT; a solver can add
    a custom frequency such as ``mujoco:actuator``. This catalog turns those
    custom rows into named, selectable objects without hard-coding actuators
    or other solver concepts into ModelTree.

    ModelTree owns one catalog per Model. The catalog maps each custom row to
    its label, world, articulation, and paged tree key. PropertyInspector uses
    that frequency and row index to find matching attributes; it owns value
    display and editing. The catalog does not fetch values for property display
    or own simulation state, and assignment to Model/State/Control is a separate
    concern handled by the attribute inspector.

    Ownership arrays are read once for this Model. An explicit articulation
    association takes precedence over a single unambiguous world reference;
    otherwise rows remain shared. Labels and paged tree nodes are materialized
    on demand. Recreate the catalog when the Model's frequency layout changes.
    """

    page_size = 128

    def __init__(self, model: Model) -> None:
        """Index nonempty custom frequencies by world and articulation.

        Read ownership arrays from the device once, retaining host index buckets
        and references to authored labels. Valid articulation ownership overrides
        a world reference; unresolved ownership uses -1 for the shared world or
        no articulation. Tree rows are created later by add_world and expand.
        """
        self.model = model
        self.counts = {name: count for name, count in model.custom_frequency_counts.items() if count > 0}
        self.worlds = {}
        self.owners = {}
        self.labels = {}
        self.buckets = defaultdict(list)
        self.articulations: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        self.groups = {}
        self.owner_groups = {}
        world_attributes = defaultdict(list)
        for name, spec in model._iter_attribute_specs():
            if (
                spec.frequency in self.counts
                and spec.assignment in (None, Model.AttributeAssignment.MODEL)
                and spec.references == Model.AttributeFrequency.WORLD
                and spec.row_width == 1
            ):
                world_attributes[spec.frequency].append(name)
        articulation_world = None
        for frequency, count in self.counts.items():
            worlds = np.full(count, -1, dtype=np.int32)
            owners = np.full(count, -1, dtype=np.int32)
            # A single explicit world reference is unambiguous. Multiple world
            # references can describe a relationship spanning worlds, not ownership.
            if len(world_attributes[frequency]) == 1:
                value = attribute_value(model, world_attributes[frequency][0])
                values = value.numpy() if isinstance(value, wp.array) else np.asarray(value)
                if values.shape == (count,) and values.dtype.kind in "iu":
                    valid = (values >= 0) & (values < model.world_count)
                    worlds[valid] = values[valid]
            value = model.custom_frequency_articulation.get(frequency)
            if value is not None:
                values = value.numpy() if isinstance(value, wp.array) else np.asarray(value)
                if values.shape == (count,) and values.dtype.kind in "iu":
                    valid = (values >= 0) & (values < model.articulation_count)
                    owners[valid] = values[valid]
                    if np.any(valid):
                        if articulation_world is None:
                            articulation_world = model.articulation_world.numpy()
                        worlds[valid] = articulation_world[owners[valid]]
            self.worlds[frequency], self.owners[frequency] = worlds, owners
            label_attribute = model.custom_frequency_label_attributes.get(frequency)
            self.labels[frequency] = attribute_value(model, label_attribute) if label_attribute else None
            order = np.lexsort((owners, worlds))
            boundaries = (
                np.flatnonzero((worlds[order[1:]] != worlds[order[:-1]]) | (owners[order[1:]] != owners[order[:-1]]))
                + 1
            )
            for indices in np.split(order, boundaries):
                first = indices[0]
                self.buckets[int(worlds[first])].append((frequency, int(owners[first]), indices))
                if owners[first] >= 0:
                    self.articulations[int(owners[first])][frequency] = indices

    @staticmethod
    def key(frequency: str, index: int) -> TreeKey:
        """Encode a frequency name and its zero-based storage row as an explorer key."""
        return "custom:" + frequency, int(index)

    @staticmethod
    def frequency(key: TreeKey) -> str | None:
        """Decode a custom-object key, returning None for other explorer node types."""
        return key[0].removeprefix("custom:") if key[0].startswith("custom:") else None

    def world_for(self, key: TreeKey) -> int:
        """Return the cached world of a valid custom-object key, or -1 for shared data."""
        return int(self.worlds[self.frequency(key)][key[1]])

    def label(self, frequency: str, index: int) -> str:
        """Return an authored nonempty string label or a frequency-and-index fallback.

        The index is the zero-based storage row, not its position within a page.
        Only host-side label containers are inspected; this does not read a
        Warp array back from the device.
        """
        labels = self.labels[frequency]
        if isinstance(labels, (list, tuple, np.ndarray)) and index < len(labels):
            label = labels[index]
            if isinstance(label, str) and label.strip():
                return label
        return f"{frequency.rsplit(':', maxsplit=1)[-1].replace('_', ' ').capitalize()} {index}"

    def add_world(self, tree: ModelTree, world: int) -> None:
        """Attach this world's frequency groups beneath their owning articulations or root.

        Called once when ModelTree materializes a world. Store group keys for
        later reference navigation; leave individual custom-object rows lazy.
        """
        for frequency, owner, indices in self.buckets.get(world, ()):
            parent = ("articulation", owner) if owner >= 0 else ("root", world)
            key = self._group(tree, parent, frequency, indices, world, f"{frequency} · {len(indices):,}")
            self.owner_groups[(frequency, world, owner)] = key

    def _group(
        self,
        tree: ModelTree,
        parent: TreeKey,
        frequency: str,
        indices: Sequence[int] | np.ndarray,
        world: int,
        label: str,
    ) -> TreeKey:
        """Register a lazy group and its storage indices, attach it to the tree, and return its key."""
        key = ("frequency", len(self.groups))
        self.groups[key] = frequency, indices
        tree.labels[key] = label
        tree.meta[key] = {"world": world}
        tree.parents[key] = parent
        tree.children[parent].append(key)
        return key

    def expand(self, tree: ModelTree, key: TreeKey) -> None:
        """Materialize one level of a custom-frequency group, leaving loaded groups unchanged.

        Split large index sets into recursively paged groups so each expansion
        adds at most page_size children. Range captions are one-based positions
        within the group; leaf keys keep their original zero-based storage rows.
        """
        if key not in self.groups or tree.children.get(key):
            return
        frequency, indices = self.groups[key]
        world = tree.meta[key]["world"]
        if len(indices) > self.page_size:
            size = self.page_size
            while len(indices) > size * self.page_size:
                size *= self.page_size
            for start in range(0, len(indices), size):
                end = min(len(indices), start + size)
                self._group(tree, key, frequency, indices[start:end], world, f"{start + 1:,}-{end:,}")
        else:
            for index in indices:
                child = self.key(frequency, index)
                tree.labels[child] = self.label(frequency, index)
                tree.meta[child] = {"world": world}
                tree.parents[child] = key
                tree.children[key].append(child)

    def reveal(self, tree: ModelTree, key: TreeKey) -> None:
        """Materialize the world and page chain containing a valid custom-object key.

        Descend only into the page containing the storage index. This creates
        parent links for navigation without expanding every page or changing
        the tree's visible expansion set.
        """
        frequency, index = self.frequency(key), key[1]
        world = self.world_for(key)
        tree.ensure_world(world)
        group = self.owner_groups[(frequency, world, int(self.owners[frequency][index]))]
        while key not in tree.parents:
            self.expand(tree, group)
            if key in tree.parents:
                break
            group = next(
                child
                for child in tree.children[group]
                if self.groups[child][1][0] <= index <= self.groups[child][1][-1]
            )

    def search(self, query: str) -> Iterator[tuple[TreeKey, str]]:
        """Yield matching custom-object keys and labels without materializing tree rows.

        The caller supplies a casefolded query. Search authored labels and full
        namespaced frequency names on the host, including collapsed worlds.
        """
        for frequency, count in self.counts.items():
            match_frequency = query in frequency.casefold()
            for index in range(count):
                label = self.label(frequency, index)
                if match_frequency or query in label.casefold():
                    yield self.key(frequency, index), label
