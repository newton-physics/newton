# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lazy model topology, resource identities, and explorer selection domains."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..sim.model import Model
from .viewer_session import ViewerSession

if TYPE_CHECKING:
    from typing import Any

    from .gui.collision import CollisionInspector
    from .viewer_types import TreeKey, ViewerExample


class ModelTree:
    """Index one Model's explorer hierarchy independently of its presentation.

    World boundary arrays are read at construction; object topology is loaded
    only for expanded or inspected worlds. Keys retain global Model indices,
    so filtering and visibility never renumber objects or their world offsets.
    Resource and custom-frequency catalogs add their own keys lazily.

    Cached topology assumes a fixed Model layout. Construct a new tree after a
    Model rebuild; ``rebuild()`` only regenerates the visible explorer rows.
    """

    def __init__(
        self,
        model: Model,
        world: int = 0,
        *,
        example: ViewerExample | None = None,
        session: ViewerSession | None = None,
    ) -> None:
        self.model = model
        self.session = session if session is not None else ViewerSession.from_example(example)
        self.children: defaultdict[TreeKey, list[TreeKey]] = defaultdict(list)
        self.labels: dict[TreeKey, str] = {}
        self.expanded: set[TreeKey] = set()
        self.rows: list[tuple[TreeKey, int]] = []
        self.query = ""
        self.revision = 0
        self.meta: dict[TreeKey, dict[str, Any]] = {}
        self.parents: dict[TreeKey, TreeKey] = {}
        self.loaded_worlds: set[int] = set()
        self._shape_descendants: dict[TreeKey, list[int]] = {}
        self._joint_descendants: dict[TreeKey, list[int]] = {}
        self._reference_names: dict[tuple[str, int], Counter[str]] = {}
        self.collision: CollisionInspector | None = None
        self.kinds = ("body", "joint", "shape", "articulation")
        from .viewer_deformables import DeformableCatalog  # noqa: PLC0415
        from .viewer_frequencies import FrequencyCatalog  # noqa: PLC0415
        from .viewer_resources import GeometryResources  # noqa: PLC0415

        self.deformables = DeformableCatalog(model, example)
        self.custom = FrequencyCatalog(model)
        self.resources = GeometryResources(model, example, session=self.session)
        # These small boundary arrays scale with worlds, not scene objects.
        self.starts: dict[str, np.ndarray] = {}
        for kind in (*self.kinds, "particle"):
            starts = getattr(model, kind + "_world_start", None)
            count = getattr(model, kind + "_count", 0)
            fallback = np.full(model.world_count + 2, count, dtype=int)
            fallback[0] = 0
            self.starts[kind] = starts.numpy() if starts is not None else fallback
        self.roots = [("root", i) for i in range(model.world_count)]
        for key in self.roots:
            self.labels[key] = f"World {key[1]}"
        if (
            not self.roots
            or any(self.ranges(kind, -1) for kind in self.starts)
            or any(source.ranges(-1) for source in self.deformables.sources)
            or -1 in self.custom.buckets
        ):
            self.roots.append(("root", -1))
            self.labels[("root", -1)] = "Shared" if model.world_count else "Scene"
        # Builders without explicit world scopes keep all objects global. In
        # that single-world case, avoid an empty World 0 beside its own contents.
        if (
            model.world_count == 1
            and ("root", -1) in self.roots
            and not any(self.ranges(kind, 0) for kind in self.starts)
            and not any(source.ranges(0) for source in self.deformables.sources)
            and 0 not in self.custom.buckets
        ):
            self.roots.remove(("root", 0))
            self.labels.pop(("root", 0))
            self.labels[("root", -1)] = "World 0"
        first = world if ("root", world) in self.roots else self.roots[0][1]
        self.ensure_world(first)
        self.expanded.add(("root", first))
        self.resources.attach(self)
        self.rebuild()

    def ranges(self, kind: str, world: int) -> list[tuple[int, int]]:
        """Return half-open global index ranges; world -1 includes shared objects."""
        starts = self.starts[kind]
        pairs = ((0, starts[0]), (starts[-2], starts[-1])) if world < 0 else ((starts[world], starts[world + 1]),)
        return [(int(a), int(b)) for a, b in pairs if b > a]

    def world_for(self, key: TreeKey) -> int:
        kind, index = key
        if "world" in self.meta.get(key, {}):
            return self.meta[key]["world"]
        if kind == "root":
            return index
        if kind == "group":
            return index // 4
        if kind in ("tri", "tet"):
            return self.world_for(("particle", int(self.deformables.topology(kind)[index, 0])))
        if kind == "particle":
            return self.deformables.world_for(index)
        if self.custom.frequency(key) is not None:
            return self.custom.world_for(key)
        starts = self.starts[kind]
        world = int(np.searchsorted(starts, index, side="right")) - 1
        return world if 0 <= world < self.model.world_count else -1

    def reference_caption(self, key: TreeKey, label: str) -> str:
        """Keep shortened reference names distinct without loading world topology."""
        kind, index = key
        caption = label.rstrip("/").rsplit("/", 1)[-1]
        if kind not in self.starts:
            return caption
        world = self.world_for(key)
        bucket = (kind, world)
        if bucket not in self._reference_names:
            labels = getattr(self.model, kind + "_label", ())
            self._reference_names[bucket] = Counter(
                labels[i].rstrip("/").rsplit("/", 1)[-1]
                for first, end in self.ranges(kind, world)
                for i in range(first, min(end, len(labels)))
                if labels[i]
            )
        if self._reference_names[bucket][caption] > 1:
            caption = f"#{index} · {caption}"
        if self.model.world_count > 1:
            owner = f"World {world}" if world >= 0 else "Shared"
            caption = f"{owner} · {caption}"
        return caption

    def ensure_world(self, world: int) -> None:
        """Load one world's topology once, without materializing other worlds.

        Device reads are restricted to this world's object ranges. Joint/body
        parentage builds the hierarchy; shapes are children of their owning body,
        while parentless bodies and static geometry receive separate groups.
        """
        if world in self.loaded_worlds:
            return
        self.loaded_worlds.add(world)
        m = self.model
        root = ("root", world)
        groups = {1: "Rigid bodies", 2: "Static geometry"}
        group_keys = {i: ("group", world * 4 + i) for i in groups}
        for i, label in groups.items():
            key = group_keys[i]
            self.children[root].append(key)
            self.labels[key] = label
        indices = {kind: [i for a, b in self.ranges(kind, world) for i in range(a, b)] for kind in self.kinds}
        bodies, articulations = set(indices["body"]), set(indices["articulation"])
        local_keys = [root, *group_keys.values()]
        for kind in self.kinds:
            labels = getattr(m, kind + "_label", [])
            for i in indices[kind]:
                key = (kind, i)
                self.labels[key] = labels[i] if i < len(labels) and labels[i] else f"{kind} {i}"
                local_keys.append(key)

        def values(kind: str, name: str) -> list[Any]:
            array = getattr(m, f"{kind}_{name}")
            return [value for a, b in self.ranges(kind, world) for value in array[a:b].numpy()]

        for kind, names in (("body", ("flags",)), ("shape", ("flags", "type", "opacity")), ("joint", ("type",))):
            for name in names:
                for i, value in zip(indices[kind], values(kind, name), strict=True):
                    self.meta.setdefault((kind, i), {})[name] = value.item()
        parented = set()
        for k, p, c, art in zip(
            indices["joint"],
            values("joint", "parent"),
            values("joint", "child"),
            values("joint", "articulation"),
            strict=True,
        ):
            key = ("joint", k)
            if p in bodies and p != c:
                owner = ("body", int(p))
            elif art in articulations:
                owner = ("articulation", int(art))
            else:
                owner = group_keys[1]
            self.children[owner].append(key)
            if c in bodies and c not in parented:
                self.children[key].append(("body", int(c)))
                parented.add(int(c))
                self.meta.setdefault(("body", int(c)), {})["joint"] = k
        for i in indices["articulation"]:
            self.children[root].append(("articulation", i))
        for i in indices["body"]:
            if i not in parented:
                self.children[group_keys[1]].append(("body", i))
        for i, body in zip(indices["shape"], values("shape", "body"), strict=True):
            owner = ("body", int(body)) if body in bodies else group_keys[2]
            self.children[owner].append(("shape", i))
        self.expanded.add(group_keys[1])
        if len(articulations) <= 4:
            self.expanded.update(("articulation", i) for i in articulations)
        # Closed kinematic loops can have no parentless body. Keep their
        # disconnected components reachable without recursively traversing them.
        reached = set()
        pending = [root]
        while pending:
            key = pending.pop()
            if key not in reached:
                reached.add(key)
                pending.extend(self.children.get(key, ()))
        for key in local_keys:
            if key in reached:
                continue
            self.children[group_keys[1]].append(key)
            pending = [key]
            while pending:
                child_key = pending.pop()
                if child_key not in reached:
                    reached.add(child_key)
                    pending.extend(self.children.get(child_key, ()))
        self.children[root] = [("articulation", i) for i in indices["articulation"]] + [
            group for group in group_keys.values() if self.children[group]
        ]
        for parent in local_keys:
            for child in self.children.get(parent, ()):
                self.parents.setdefault(child, parent)
        self.deformables.add_world(self, world)
        self.custom.add_world(self, world)

    def has_children(self, key: TreeKey) -> bool:
        if key[0] == "shape" or self.resources.handles(key):
            return self.resources.has_children(self, key)
        if self.collision is not None and (key == self.collision.root or key in self.collision.groups):
            return True
        if key[0] == "root" and key[1] not in self.loaded_worlds:
            return any(self.ranges(kind, key[1]) for kind in self.starts) or key[1] in self.custom.buckets
        if key in self.deformables.groups or key in self.custom.groups:
            return True
        return bool(self.children.get(key))

    def reveal_attribute(self, key: TreeKey) -> None:
        """Keep referenced domains without spatial tree nodes inspectable as rows."""
        if key in self.parents:
            return
        frequency = Model.AttributeFrequency[key[0].split(":", 1)[1]]
        world = -1
        # Only explicit world references assign a non-spatial row to a world.
        candidates = [
            name
            for name, spec in self.model._iter_attribute_specs()
            if spec.frequency == frequency
            and spec.references == Model.AttributeFrequency.WORLD
            and spec.assignment in (None, Model.AttributeAssignment.MODEL)
            and spec.row_width == 1
        ]
        if len(candidates) == 1:
            from .viewer_frequencies import attribute_value  # noqa: PLC0415

            array = attribute_value(self.model, candidates[0])
            if isinstance(array, wp.array) and array.ndim == 1 and key[1] < len(array):
                value = int(array[key[1] : key[1] + 1].numpy()[0])
                if 0 <= value < self.model.world_count:
                    world = value
        root = ("root", world)
        if root not in self.roots:
            self.roots.append(root)
            self.labels[root] = "Shared" if world < 0 else f"World {world}"
        self.ensure_world(world)
        group = ("reference_group", (world, int(frequency)))
        if group not in self.parents:
            self.parents[group] = root
            self.labels[group] = frequency.name.replace("_", " ").title()
            self.meta[group] = {"world": world, "frequency": frequency}
            self.children[root].append(group)
        self.parents[key] = group
        self.children[group].append(key)
        self.meta[key] = {"world": world}
        self.labels[key] = f"{frequency.name.replace('_', ' ').title()} {key[1]}"

    def rebuild(self) -> None:
        self.revision += 1
        query = self.query.casefold().strip()
        if query:
            if query.startswith("sdf ") and query[4:].isdigit():
                self.resources.reveal(self, ("sdf", int(query[4:])))
            # Labels already live on the CPU. Searching collapsed worlds does
            # not pull their topology or state back from the device.
            matches = defaultdict(list)
            for kind in self.kinds:
                for i, name in enumerate(getattr(self.model, kind + "_label", ())):
                    label = name or f"{kind} {i}"
                    if query in label.casefold():
                        key = (kind, i)
                        self.labels[key] = label
                        matches[self.world_for(key)].append((key, 1))
            self.rows = []
            for key, label in self.custom.search(query):
                self.labels[key] = label
                matches[self.world_for(key)].append((key, 1))
            for _kind, name, world, *_ranges in getattr(self.model, "_deformable_objects", ()):
                if query in name.casefold():
                    self.ensure_world(world)
            parts = query.split()
            if len(parts) == 2 and parts[1].isdigit():
                kind = {"particle": "particle", "triangle": "tri", "tetrahedron": "tet", "tet": "tet"}.get(parts[0])
                if kind is not None and int(parts[1]) < getattr(self.model, kind + "_count"):
                    self.deformables.reveal(self, (kind, int(parts[1])))
            for key, name in self.labels.items():
                if (
                    key[0] in ("surface", "volume", "particle_set", "particle", "tri", "tet")
                    or self.resources.shape_owner(key) is not None
                ) and query in name.casefold():
                    matches[self.world_for(key)].append((key, 1))
            for root in self.roots:
                if self.resources.handles(root):
                    resource_rows = [
                        (key, 1)
                        for key, label in self.labels.items()
                        if key != root
                        and self.resources.handles(key)
                        and self.resources.bvhs.handles(key) == self.resources.bvhs.handles(root)
                        and self.resources.shape_owner(key) is None
                        and query in label.casefold()
                    ]
                    if resource_rows or query in self.labels[root].casefold():
                        self.rows.append((root, 0))
                        self.rows.extend(resource_rows)
                    continue
                if self.collision is not None and self.collision.handles(root):
                    matches_collision = [
                        (key, 1)
                        for key, label in self.labels.items()
                        if key != root and self.collision.handles(key) and query in label.casefold()
                    ]
                    if matches_collision or query in self.labels[root].casefold():
                        self.rows.append((root, 0))
                        self.rows.extend(matches_collision)
                    continue
                if root[1] in matches or query in self.labels[root].casefold():
                    self.rows.append((root, 0))
                    self.rows.extend(matches[root[1]])
            return
        rows = []
        stack = [(root, 0) for root in reversed(self.roots)]
        seen = set()
        while stack:
            key, depth = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            rows.append((key, depth))
            if key in self.expanded:
                stack.extend((child, depth + 1) for child in reversed(self.children[key]))
        self.rows = rows

    def toggle(self, key: TreeKey) -> None:
        if key in self.expanded:
            self.expanded.remove(key)
        else:
            if key[0] == "root":
                self.ensure_world(key[1])
            self.deformables.expand(self, key)
            self.custom.expand(self, key)
            self.resources.expand(self, key)
            if self.collision is not None:
                self.collision.expand(key)
            self.expanded.add(key)
        self.rebuild()

    def shape_descendants(self, key: TreeKey) -> list[int]:
        return self._descendants(key, "shape", self._shape_descendants)

    def joint_descendants(self, key: TreeKey) -> list[int]:
        return self._descendants(key, "joint", self._joint_descendants)

    def deformable_descendants(self, key: TreeKey) -> list[TreeKey]:
        world = self.world_for(key)
        self.ensure_world(world)
        if key[0] == "root" or (key[0] == "group" and key[1] % 4 == 3):
            return [(obj.kind, i) for i, obj in enumerate(self.deformables.objects) if obj.world == world]
        return []

    def _descendants(self, key: TreeKey, kind: str, cache: dict[TreeKey, list[int]]) -> list[int]:
        self.ensure_world(self.world_for(key))
        if key not in cache:
            result, seen, pending = [], set(), [key]
            while pending:
                item = pending.pop()
                if item in seen:
                    continue
                seen.add(item)
                if item[0] == kind:
                    result.append(item[1])
                pending.extend(self.children.get(item, ()))
            cache[key] = result
        return cache[key]
