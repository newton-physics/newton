# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lazy model topology, resource identities, and explorer selection domains."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..sim.model import Model
from .viewer_bindings import InspectionBindings

if TYPE_CHECKING:
    from typing import Any

    from .gui.collision import CollisionInspector
    from .viewer_types import TreeKey


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
        bindings: InspectionBindings | None = None,
    ) -> None:
        """Create resource catalogs and world roots, materializing only the initial world.

        Cache world boundary arrays and ownership metadata on the host. Object
        topology for other worlds remains lazy; the bindings supply inspected
        solver resources without transferring live state here.
        """
        self.model = model
        self.bindings = bindings if bindings is not None else InspectionBindings()
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
        self._tendon_worlds: np.ndarray | None = None
        self._tendon_starts: np.ndarray | None = None
        self._attachment_worlds: np.ndarray | None = None
        self._mimic_worlds: np.ndarray | None = None
        self.collision: CollisionInspector | None = None
        self.kinds = ("body", "joint", "shape", "articulation")
        from .viewer_deformables import DeformableCatalog  # noqa: PLC0415
        from .viewer_frequencies import CustomFrequencyIndex  # noqa: PLC0415
        from .viewer_resources import GeometryInspector  # noqa: PLC0415

        self.deformables = DeformableCatalog(model)
        self.custom = CustomFrequencyIndex(model)
        self.resources = GeometryInspector(model, bindings=self.bindings)
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
        if not self.roots or self._world_has_content(-1):
            self.roots.append(("root", -1))
            self.labels[("root", -1)] = "Shared" if model.world_count else "Scene"
        # Builders without explicit world scopes keep all objects global. In
        # that single-world case, avoid an empty World 0 beside its own contents.
        if model.world_count == 1 and ("root", -1) in self.roots and not self._world_has_content(0):
            self.roots.remove(("root", 0))
            self.labels.pop(("root", 0))
            self.labels[("root", -1)] = "World 0"
        first = world if ("root", world) in self.roots else self.roots[0][1]
        self.ensure_world(first)
        self.expanded.add(("root", first))
        self._attach_actuators()
        self.resources.attach(self)
        self.rebuild()

    def _tendon_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Cache host tendon world and guide-start arrays once when tendons exist."""
        if self._tendon_worlds is None:
            m = self.model
            if m.tendon_count and m.tendon_world is not None and m.tendon_start is not None:
                self._tendon_worlds = m.tendon_world.numpy()
                self._tendon_starts = m.tendon_start.numpy()
            else:
                self._tendon_worlds = np.empty(0, dtype=np.int32)
                self._tendon_starts = np.empty(0, dtype=np.int32)
        return self._tendon_worlds, self._tendon_starts

    def tendon_elements(self, index: int) -> tuple[range, range]:
        """Return (guide_range, segment_range) for a valid tendon index."""
        _, starts = self._tendon_arrays()
        if 0 <= index < len(starts) - 1:
            g0, g1 = int(starts[index]), int(starts[index + 1])
            s0 = g0 - index
            s1 = max(s0, g1 - index - 1)
            return range(g0, g1), range(s0, s1)
        return range(0), range(0)

    def tendon_for_element(self, kind: str, index: int) -> int:
        """Resolve a tendon_guide or tendon_segment index back to its owning tendon index, or -1."""
        _, starts = self._tendon_arrays()
        if len(starts) <= 1 or index < 0:
            return -1
        if kind == "tendon_guide":
            tendon = int(np.searchsorted(starts, index, side="right")) - 1
            return tendon if 0 <= tendon < len(starts) - 1 else -1
        if kind == "tendon_segment":
            seg_starts = starts - np.arange(len(starts), dtype=starts.dtype)
            tendon = int(np.searchsorted(seg_starts, index, side="right")) - 1
            return tendon if 0 <= tendon < len(starts) - 1 else -1
        return -1

    def tendons_in_world(self, world: int) -> list[int]:
        """Return tendon indices owned by a world (or -1 for global tendons)."""
        worlds, _ = self._tendon_arrays()
        if not len(worlds):
            return []
        if world < 0:
            return [int(i) for i in np.flatnonzero((worlds < 0) | (worlds >= self.model.world_count))]
        return [int(i) for i in np.flatnonzero(worlds == world)]

    def _attachment_world_array(self) -> np.ndarray:
        """Cache host particle-body attachment world indices once when attachments exist."""
        if self._attachment_worlds is None:
            m = self.model
            if m.attachment_particle_body_count and m.attachment_particle_body_world is not None:
                self._attachment_worlds = m.attachment_particle_body_world.numpy()
            else:
                self._attachment_worlds = np.empty(0, dtype=np.int32)
        return self._attachment_worlds

    def attachments_in_world(self, world: int) -> list[int]:
        """Return particle-body attachment indices owned by a world (or -1 for global attachments)."""
        worlds = self._attachment_world_array()
        if not len(worlds):
            return []
        if world < 0:
            return [int(i) for i in np.flatnonzero((worlds < 0) | (worlds >= self.model.world_count))]
        return [int(i) for i in np.flatnonzero(worlds == world)]

    def _mimic_world_array(self) -> np.ndarray:
        """Cache host mimic-constraint world indices once when mimic constraints exist."""
        if self._mimic_worlds is None:
            m = self.model
            if m.constraint_mimic_count and m.constraint_mimic_world is not None:
                self._mimic_worlds = m.constraint_mimic_world.numpy()
            else:
                self._mimic_worlds = np.empty(0, dtype=np.int32)
        return self._mimic_worlds

    def mimics_in_world(self, world: int) -> list[int]:
        """Return mimic-constraint indices owned by a world (or -1 for global constraints)."""
        worlds = self._mimic_world_array()
        if not len(worlds):
            return []
        if world < 0:
            return [int(i) for i in np.flatnonzero((worlds < 0) | (worlds >= self.model.world_count))]
        return [int(i) for i in np.flatnonzero(worlds == world)]

    def _world_has_content(self, world: int) -> bool:
        """Report whether a world owns any spatial, deformable, tendon, attachment, mimic, or custom objects."""
        return (
            any(self.ranges(kind, world) for kind in self.starts)
            or self.deformables.has_world(world)
            or world in self.custom.buckets
            or bool(self.tendons_in_world(world))
            or bool(self.attachments_in_world(world))
            or bool(self.mimics_in_world(world))
        )

    def _attach_actuators(self) -> None:
        """Attach a top-level Actuators/Batteries collection when the model owns any."""
        act_count = len(self.model.actuators)
        bat_count = len(self.model.batteries)
        if not (act_count or bat_count):
            return
        root = ("actuator_collection", 0)
        if act_count and bat_count:
            label = f"Actuators · {act_count:,} & Batteries · {bat_count:,}"
        elif act_count:
            label = f"Actuators · {act_count:,}"
        else:
            label = f"Batteries · {bat_count:,}"
        self.roots.append(root)
        self.labels[root] = label
        self.meta[root] = {"world": -1}

    def _expand_aux(self, key: TreeKey) -> None:
        """Lazily populate tendon children, attachment pages, and actuator/battery collections."""
        kind, index = key
        if self.children.get(key):
            return
        if kind == "tendon":
            world = self.world_for(key)
            guides, segments = self.tendon_elements(index)
            for g in guides:
                child = ("tendon_guide", g)
                self.labels[child] = f"Guide {g}"
                self.meta[child] = {"world": world, "tendon": index}
                self.parents[child] = key
                self.children[key].append(child)
            for s in segments:
                child = ("tendon_segment", s)
                self.labels[child] = f"Segment {s}"
                self.meta[child] = {"world": world, "tendon": index}
                self.parents[child] = key
                self.children[key].append(child)
        elif kind == "attachment_group":
            indices = self.attachments_in_world(index)
            page_size = 128
            if len(indices) > page_size:
                for start in range(0, len(indices), page_size):
                    end = min(len(indices), start + page_size)
                    page = ("attachment_page", (index, start, end))
                    self.labels[page] = f"{start + 1:,}-{end:,}"
                    self.meta[page] = {"world": index, "indices": indices[start:end]}
                    self.parents[page] = key
                    self.children[key].append(page)
            else:
                for idx in indices:
                    child = ("attachment_particle_body", idx)
                    self.labels[child] = f"Attachment {idx}"
                    self.meta[child] = {"world": index}
                    self.parents[child] = key
                    self.children[key].append(child)
        elif kind == "attachment_page":
            world = self.meta[key]["world"]
            for idx in self.meta[key]["indices"]:
                child = ("attachment_particle_body", idx)
                self.labels[child] = f"Attachment {idx}"
                self.meta[child] = {"world": world}
                self.parents[child] = key
                self.children[key].append(child)
        elif kind == "actuator_collection":
            for i, actuator in enumerate(self.model.actuators):
                child = ("actuator_group", i)
                drive = type(actuator.drive).__name__
                self.labels[child] = f"Actuator {i} · {drive} ({actuator.num_actuators:,})"
                self.meta[child] = {"world": -1}
                self.parents[child] = key
                self.children[key].append(child)
            for i, battery in enumerate(self.model.batteries):
                child = ("battery", i)
                self.labels[child] = f"Battery {i} · {type(battery).__name__} ({battery.num_batteries:,})"
                self.meta[child] = {"world": -1}
                self.parents[child] = key
                self.children[key].append(child)

    def reveal_aux(self, key: TreeKey) -> None:
        """Materialize ancestor nodes for a tendon element, attachment, or actuator/battery key."""
        kind, index = key
        if kind in ("tendon_guide", "tendon_segment"):
            tendon = self.tendon_for_element(kind, index)
            if tendon >= 0:
                self.ensure_world(self.world_for(("tendon", tendon)))
                self._expand_aux(("tendon", tendon))
        elif kind == "attachment_particle_body":
            world = self.world_for(key)
            self.ensure_world(world)
            group = ("attachment_group", world)
            self._expand_aux(group)
            for child in self.children.get(group, ()):
                if child[0] == "attachment_page" and index in self.meta[child].get("indices", ()):
                    self._expand_aux(child)
                    break
        elif kind in ("actuator_group", "battery"):
            self._expand_aux(("actuator_collection", 0))

    def ranges(self, kind: str, world: int) -> list[tuple[int, int]]:
        """Return half-open global index ranges; world -1 includes shared objects."""
        starts = self.starts[kind]
        pairs = ((0, starts[0]), (starts[-2], starts[-1])) if world < 0 else ((starts[world], starts[world + 1]),)
        return [(int(a), int(b)) for a, b in pairs if b > a]

    def world_for(self, key: TreeKey) -> int:
        """Resolve a valid object key to its world index, using -1 for shared objects."""
        kind, index = key
        if "world" in self.meta.get(key, {}):
            return self.meta[key]["world"]
        if kind in ("root", "tendon_group", "attachment_group", "mimic_group"):
            return index
        if kind == "attachment_page":
            return index[0]
        if kind == "group":
            return index // 4
        if kind in ("actuator_collection", "actuator_group", "battery"):
            return -1
        if kind == "curve":
            return self.deformables.curve_world(index)
        if kind == "deformable_visual_mesh":
            meshes = self.model.deformable_visual_meshes
            return meshes[index].world if 0 <= index < len(meshes) else -1
        if kind == "edge":
            return self.world_for(("particle", int(self.deformables.topology("edge")[index, 2])))
        if kind in ("tri", "tet"):
            return self.world_for(("particle", int(self.deformables.topology(kind)[index, 0])))
        if kind == "particle":
            return self.deformables.world_for(index)
        if kind == "tendon":
            worlds, _ = self._tendon_arrays()
            return int(worlds[index]) if 0 <= index < len(worlds) else -1
        if kind in ("tendon_guide", "tendon_segment"):
            tendon = self.tendon_for_element(kind, index)
            return self.world_for(("tendon", tendon)) if tendon >= 0 else -1
        if kind == "attachment_particle_body":
            worlds = self._attachment_world_array()
            return int(worlds[index]) if 0 <= index < len(worlds) else -1
        if kind == "constraint_mimic":
            worlds = self._mimic_world_array()
            return int(worlds[index]) if 0 <= index < len(worlds) else -1
        if self.custom.frequency(key) is not None:
            return self.custom.world_for(key)
        if kind not in self.starts:
            return -1
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
        tendons = self.tendons_in_world(world)
        if tendons:
            tgroup = ("tendon_group", world)
            self.labels[tgroup] = f"Tendons · {len(tendons):,}"
            self.meta[tgroup] = {"world": world}
            self.parents[tgroup] = root
            self.children[root].append(tgroup)
            labels = m.tendon_label
            for i in tendons:
                tkey = ("tendon", i)
                self.labels[tkey] = labels[i] if i < len(labels) and labels[i] else f"Tendon {i}"
                self.meta[tkey] = {"world": world}
                self.parents[tkey] = tgroup
                self.children[tgroup].append(tkey)
        attachments = self.attachments_in_world(world)
        if attachments:
            agroup = ("attachment_group", world)
            self.labels[agroup] = f"Attachments · {len(attachments):,}"
            self.meta[agroup] = {"world": world}
            self.parents[agroup] = root
            self.children[root].append(agroup)
        mimics = self.mimics_in_world(world)
        if mimics:
            mgroup = ("mimic_group", world)
            self.labels[mgroup] = f"Mimic constraints · {len(mimics):,}"
            self.meta[mgroup] = {"world": world}
            self.parents[mgroup] = root
            self.children[root].append(mgroup)
            labels = m.constraint_mimic_label
            for i in mimics:
                mkey = ("constraint_mimic", i)
                self.labels[mkey] = labels[i] if i < len(labels) and labels[i] else f"Mimic {i}"
                self.meta[mkey] = {"world": world}
                self.parents[mkey] = mgroup
                self.children[mgroup].append(mkey)
        self.custom.add_world(self, world)

    def has_children(self, key: TreeKey) -> bool:
        """Report whether a node is expandable, including children not yet materialized."""
        if key[0] == "shape" or self.resources.handles(key):
            return self.resources.has_children(self, key)
        if self.collision is not None and (key == self.collision.root or key in self.collision.groups):
            return True
        if key[0] == "root" and key[1] not in self.loaded_worlds:
            return self._world_has_content(key[1])
        if key[0] in ("tendon", "attachment_group", "attachment_page", "actuator_collection"):
            return True
        if key in self.deformables.groups or key in self.custom.groups:
            return True
        return bool(self.children.get(key))

    def refresh_shapes(self) -> set[TreeKey]:
        """Update shape metadata and resources while retaining explorer topology.

        Body variants can change geometry without changing object indices.
        Read metadata only for worlds already materialized by the explorer;
        resource payloads remain lazy. Return vanished resource keys so the
        presenter can redirect an obsolete resource selection to its shape.
        """
        for world in self.loaded_worlds:
            for first, end in self.ranges("shape", world):
                for name in ("flags", "type", "opacity"):
                    values = getattr(self.model, "shape_" + name)[first:end].numpy()
                    for shape, value in enumerate(values, first):
                        self.meta[("shape", shape)][name] = value.item()
        removed = self.resources.refresh_shapes(self)
        self.rebuild()
        return removed

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
        """Regenerate visible (key, depth) rows from the expansion set or search query.

        Increment the row revision without rebuilding cached Model topology.
        Name searches inspect host labels; explicit resource or element queries
        may materialize the referenced world and its containing groups.
        """
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
            for label_attr, records_fn in (
                ("surface_label", self.deformables.surface_records),
                ("volume_label", self.deformables.volume_records),
                ("curve_label", self.deformables.curve_records),
            ):
                labels = getattr(self.model, label_attr, ())
                if any(query in (name or "").casefold() for name in labels):
                    for record in records_fn():
                        if query in record[1].casefold():
                            self.ensure_world(record[2])
            for mesh in self.model.deformable_visual_meshes:
                name = mesh.label or f"Skinned mesh {mesh.index}"
                if query in name.casefold():
                    self.ensure_world(mesh.world)
            for i, name in enumerate(self.model.tendon_label):
                label = name or f"Tendon {i}"
                if query in label.casefold():
                    self.ensure_world(self.world_for(("tendon", i)))
            for i, name in enumerate(self.model.constraint_mimic_label):
                label = name or f"Mimic {i}"
                if query in label.casefold():
                    self.ensure_world(self.world_for(("constraint_mimic", i)))
            parts = query.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                kind = {
                    "particle": "particle",
                    "triangle": "tri",
                    "edge": "edge",
                    "tetrahedron": "tet",
                    "tet": "tet",
                }.get(parts[0])
                if kind is not None and idx < getattr(self.model, kind + "_count"):
                    self.deformables.reveal(self, (kind, idx))
                elif parts[0] == "guide" and idx < self.model.tendon_guide_count:
                    self.reveal_aux(("tendon_guide", idx))
                elif parts[0] == "segment" and idx < self.model.tendon_segment_count:
                    self.reveal_aux(("tendon_segment", idx))
                elif parts[0] == "attachment" and idx < self.model.attachment_particle_body_count:
                    self.reveal_aux(("attachment_particle_body", idx))
            for key, name in self.labels.items():
                if (
                    key[0]
                    in (
                        "surface",
                        "volume",
                        "curve",
                        "deformable_visual_mesh",
                        "particle_set",
                        "particle",
                        "tri",
                        "edge",
                        "tet",
                        "tendon",
                        "tendon_guide",
                        "tendon_segment",
                        "attachment_particle_body",
                        "constraint_mimic",
                    )
                    or self.resources.shape_owner(key) is not None
                ) and query in name.casefold():
                    matches[self.world_for(key)].append((key, 1))
            for root in self.roots:
                if root[0] == "actuator_collection":
                    self._expand_aux(root)
                    actuator_rows = [
                        (child, 1) for child in self.children.get(root, ()) if query in self.labels[child].casefold()
                    ]
                    if actuator_rows or query in self.labels[root].casefold():
                        self.rows.append((root, 0))
                        self.rows.extend(actuator_rows)
                    continue
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
        """Toggle a node, lazily populate its children when opening, and rebuild visible rows."""
        if key in self.expanded:
            self.expanded.remove(key)
        else:
            if key[0] == "root":
                self.ensure_world(key[1])
            self._expand_aux(key)
            self.deformables.expand(self, key)
            self.custom.expand(self, key)
            self.resources.expand(self, key)
            if self.collision is not None:
                self.collision.expand(key)
            self.expanded.add(key)
        self.rebuild()

    def shape_descendants(self, key: TreeKey) -> list[int]:
        """Return cached shape indices beneath a key, including the key itself if it is a shape."""
        if key[0] == "curve":
            if key not in self._shape_descendants:
                bodies, _ = self.deformables.curve_elements(key[1])
                shapes: list[int] = []
                for body in bodies:
                    shapes.extend(self.model.body_shapes.get(body, ()))
                self._shape_descendants[key] = sorted(set(shapes))
            return self._shape_descendants[key]
        return self._descendants(key, "shape", self._shape_descendants)

    def joint_descendants(self, key: TreeKey) -> list[int]:
        """Return cached joint indices beneath a key, including the key itself if it is a joint."""
        if key[0] == "curve":
            if key not in self._joint_descendants:
                _, joints = self.deformables.curve_elements(key[1])
                self._joint_descendants[key] = list(joints)
            return self._joint_descendants[key]
        return self._descendants(key, "joint", self._joint_descendants)

    def deformable_descendants(self, key: TreeKey) -> list[TreeKey]:
        """Return deformable-object keys for a world or its deformables group; otherwise return none."""
        world = self.world_for(key)
        self.ensure_world(world)
        if key[0] == "root" or (key[0] == "group" and key[1] % 4 == 3):
            return [(obj.kind, i) for i, obj in enumerate(self.deformables.objects) if obj.world == world]
        return []

    def _descendants(self, key: TreeKey, kind: str, cache: dict[TreeKey, list[int]]) -> list[int]:
        """Cache a deduplicated traversal for one object kind after loading the owning world.

        Include the starting key when it matches kind. The returned list belongs
        to the topology cache and must not be mutated by callers.
        """
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
