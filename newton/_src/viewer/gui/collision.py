# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""On-demand inspection of collision outputs and the buffers that produced them."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...sim.collide import CollisionPipeline
from ...sim.contacts import Contacts
from ...sim.model import Model
from ..viewer_attributes import Assignment, AttributeRows, Frequency, attribute_summary, resolve_attribute

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from newton._src.viewer.gui.workspace import ViewerWorkspace
    from newton._src.viewer.viewer_model_tree import ModelTree
    from newton._src.viewer.viewer_types import TreeKey

    from .properties import PropertyInspector


@wp.kernel
def _rigid_label_points(
    rows: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    offset0: wp.array[wp.vec3],
    offset1: wp.array[wp.vec3],
    world_offsets: wp.array[wp.vec3],
    visible_worlds: wp.array[int],
    layer: wp.transform,
    output: wp.array2d[wp.vec3],
) -> None:
    i, side = wp.tid()
    row = rows[i]
    a, b = shape0[row], shape1[row]
    world = shape_world[a] if shape_world[a] >= 0 else shape_world[b]
    if a == b or (world >= 0 and visible_worlds and visible_worlds[world] == 0):
        output[i, side] = wp.vec3(wp.nan)
        return
    shape = a if side == 0 else b
    point = point0[row] + offset0[row] if side == 0 else point1[row] + offset1[row]
    body = shape_body[shape]
    if body >= 0:
        point = wp.transform_point(body_q[body], point)
    if world >= 0 and world_offsets:
        point += world_offsets[world]
    output[i, side] = wp.transform_point(layer, point)


class CollisionInspector:
    """Inspect transient contact buffers and pipeline stages on demand.

    Tree availability uses Python bindings only. Detailed rows are read while
    paused and cached for that inspection snapshot; contact indices identify
    buffer slots, not persistent physical contacts. Changing frame or resuming
    clears slot selection and dependent labels. Candidate pairs are available
    only at the live frame because history does not record broadphase buffers.
    """

    root = ("collision", 0)
    page_size = 128

    def __init__(self, workspace: ViewerWorkspace) -> None:
        """Initialize lazy contact snapshots, pipeline discovery, and viewport-label caches for the workspace."""
        self.workspace = workspace
        self.tree = None
        self.token = None
        self.contacts = None
        self.state = None
        self.pipelines = []
        self.counts = (0, 0)
        self.raw_counts = (0, 0)
        self.groups = {}
        self.cache = {}
        self.active_contact = None
        self.visible_contacts = ()
        self.contacts_tab = False
        self.list_owner = None
        self._overlay_key = None

    @staticmethod
    def handles(key: TreeKey) -> bool:
        """Identify pipeline, candidate-pair, and contact keys routed to this inspector."""
        return key is not None and (key[0].startswith("collision") or key[0] in ("contact", "soft_contact"))

    def _node(self, key: TreeKey, label: str, parent: TreeKey | None = None) -> None:
        """Register a collision node and append its parent link to the current tree."""
        tree = self.tree
        tree.labels[key] = label
        tree.meta[key] = {"world": -1}
        if parent is not None:
            tree.parents[key] = parent
            tree.children[parent].append(key)

    def attach(self, tree: ModelTree) -> None:
        """Synchronize the Collision root with available Python objects without reading device data.

        When sources disappear, remove collision nodes, caches, and any active
        collision selection. A new tree also resets contact-tab ownership.
        """
        if self.tree is not tree:
            self.tree, self.token = tree, None
            self.contacts_tab = False
            self.list_owner = None
            tree.collision = self
        # Discover Python objects only; availability must never read device data.
        available = self._contacts_buffer() is not None or bool(self._sources())
        if available == (self.root in tree.labels):
            return
        self.token = None
        if available:
            self._node(self.root, "Collision")
            tree.roots.append(self.root)
        else:
            for key in tuple(tree.labels):
                if self.handles(key):
                    tree.labels.pop(key, None)
                    tree.parents.pop(key, None)
                    tree.meta.pop(key, None)
                    tree.children.pop(key, None)
                    tree.expanded.discard(key)
            tree.roots.remove(self.root)
            self.groups.clear()
            self.cache.clear()
            self.contacts = self.state = None
            self.pipelines.clear()
            self.counts = self.raw_counts = (0, 0)
            self.active_contact = None
            self.visible_contacts = ()
            self.list_owner = None
            self.contacts_tab = False
            if self.handles(self.workspace.selected):
                self.workspace.select(None)
        tree.rebuild()

    def _contacts_buffer(self) -> Contacts | None:
        """Prefer explicitly logged Contacts over the example fallback, or return None."""
        ws = self.workspace
        logged = getattr(ws.viewer, "_last_contacts", None)
        bound = ws.bindings.contacts
        for value in (logged, bound):
            if isinstance(value, Contacts):
                return value
        return None

    def _sources(self) -> list[tuple[str, CollisionPipeline]]:
        """Resolve collision pipelines associated with the workspace Model through its bindings."""
        ws = self.workspace
        return ws.bindings.collision_pipelines(ws.model)

    def read(self, owner: object, name: str, start: int = 0, count: int = 1) -> np.ndarray:
        """Cache a host copy of an array slice until the contact snapshot is invalidated."""
        key = id(owner), name, start, count
        if key not in self.cache:
            array = getattr(owner, name)
            self.cache[key] = array[start : start + count].numpy() if count else np.empty(0)
        return self.cache[key]

    def sync(self, *, force: bool = False) -> None:
        """Refresh collision rows when inspection context changes or Refresh is requested.

        The token tracks Model, pause state, contact storage, and historical
        cursor. Running mode exposes settings without downloading contact rows.
        ``force`` bypasses both the interest check and token cache, for changes
        to a paused buffer that do not advance viewer time.
        """
        ws, tree = self.workspace, self.tree
        if tree is None or self.root not in tree.labels:
            return
        interested = self.root in tree.expanded or self.handles(ws.selected) or self.contacts_tab
        if not interested and not force:
            return
        timeline = getattr(ws.example, "timeline", ws.history)
        contacts = self._contacts_buffer()
        paused = ws.viewer.is_paused()
        token = (
            id(ws.model),
            paused,
            ws.viewer.time if paused else None,
            id(contacts) if paused else None,
            getattr(timeline, "index", None) if paused else None,
        )
        if token == self.token and not force:
            return
        # Contact slots are transient. Never carry a selected slot into another
        # collision pass or historical frame and imply it is the same contact.
        if ws.selected and ws.selected[0] in ("contact", "soft_contact", "collision_pair"):
            ws.select(self.root)
        self.token = token
        self.cache.clear()
        self.active_contact = None
        self.visible_contacts = ()
        self._overlay_key = None
        renderer = getattr(ws.viewer, "renderer", None)
        if renderer is not None:
            renderer._contact_labels = ()
            renderer._contact_selection = None
        expanded = tree.expanded.copy()
        for key in tuple(tree.labels):
            if self.handles(key) and key != self.root:
                tree.labels.pop(key, None)
                tree.parents.pop(key, None)
                tree.meta.pop(key, None)
                tree.children.pop(key, None)
                tree.expanded.discard(key)
        tree.children[self.root].clear()
        self.groups.clear()
        self.contacts = contacts
        self.state = getattr(ws.viewer, "_last_contact_state", None) or getattr(ws.viewer, "_last_state", None)
        self.pipelines = self._sources()
        self.counts = self.raw_counts = (0, 0)
        self._node(("collision_contacts", 0), "Contacts" if paused else "Contacts · pause to inspect", self.root)
        if paused and contacts is not None:
            counters = self.read(contacts, "contact_counters", count=2)
            self.raw_counts = tuple(max(0, int(v)) for v in counters)
            self.counts = tuple(
                min(n, capacity)
                for n, capacity in zip(
                    self.raw_counts, (contacts.rigid_contact_max, contacts.soft_contact_max), strict=True
                )
            )
            for kind, count in zip(("contact", "soft_contact"), self.counts, strict=True):
                label = "Rigid contacts" if kind == "contact" else "Soft-rigid contacts"
                key = ("collision_group", kind)
                self._node(key, f"{label} · {count:,}", ("collision_contacts", 0))
                self.groups[key] = kind, 0, count, None
        for i, (name, pipeline) in enumerate(self.pipelines):
            parent = ("collision_pipeline", i)
            self._node(parent, name.replace("_", " ").capitalize(), self.root)
            mode = {"nxn": "All pairs", "sap": "Sweep and prune", "explicit": "Explicit pairs"}[
                pipeline.broad_phase_mode
            ]
            self._node(("collision_broad", i), f"Broadphase · {mode}", parent)
            self._node(("collision_narrow", i), "Narrowphase", parent)
            if paused and ws._at_live_frame():
                count = max(0, int(self.read(pipeline, "broad_phase_pair_count")[0]))
                count = min(count, len(pipeline.broad_phase_shape_pairs))
                key = ("collision_pairs", i)
                self._node(key, f"Candidate pairs · {count:,}", ("collision_broad", i))
                self.groups[key] = "collision_pair", 0, count, i
        tree.expanded.update(key for key in expanded if key in tree.labels)
        for key in tuple(tree.expanded):
            if key in self.groups:
                self.expand(key)
        if self.handles(ws.selected) and ws.selected not in tree.labels:
            ws.select(self.root)
        tree.rebuild()

    def expand(self, key: TreeKey) -> None:
        """Refresh the collision root or lazily populate a bounded page of contacts or candidate pairs."""
        if key == self.root:
            self.sync(force=True)
        if key not in self.groups or self.tree.children.get(key):
            return
        kind, start, end, pipeline = self.groups[key]
        if end - start > self.page_size:
            size = self.page_size
            while end - start > size * self.page_size:
                size *= self.page_size
            for a in range(start, end, size):
                b = min(end, a + size)
                child = ("collision_page", (kind, pipeline, a, b))
                self.groups[child] = kind, a, b, pipeline
                self._node(child, f"{a:,}-{b - 1:,}", key)
        else:
            for i in range(start, end):
                child = (kind, (pipeline, i) if kind == "collision_pair" else i)
                label = f"Pair {i}" if pipeline is not None else f"Contact #{self.number(child)}"
                self._node(child, label, key)

    def reveal(self, key: TreeKey) -> None:
        """Materialize and expand the ancestor pages containing a contact or candidate pair."""
        pair = key[0] == "collision_pair"
        parent = ("collision_pairs", key[1][0]) if pair else ("collision_group", key[0])
        index = key[1][1] if pair else key[1]
        while parent in self.groups:
            self.expand(parent)
            self.tree.expanded.add(parent)
            pages = [p for p in self.tree.children[parent] if p in self.groups]
            if not pages:
                break
            parent = next(p for p in pages if self.groups[p][1] <= index < self.groups[p][2])
        while parent in self.tree.parents:
            parent = self.tree.parents[parent]
            self.tree.expanded.add(parent)

    def focus_points(self) -> list[np.ndarray]:
        """Use the active contact row even when its owning body/shape stays selected."""
        if not self.workspace.viewer.is_paused():
            return []
        self.sync()
        key = self.workspace.selected
        if key and key[0] in ("contact", "soft_contact"):
            return self.points(key)
        if self.contacts_tab and key == self.list_owner and self.active_contact is not None:
            return self.points(self.active_contact)
        return []

    def shapes(self, key: TreeKey) -> tuple[int, ...]:
        """Read shapes referenced by a paused contact or pair; return an empty tuple during playback."""
        if not self.workspace.viewer.is_paused():
            return ()
        kind, index = key
        if kind == "collision_pair":
            pipeline, row = index
            return tuple(int(v) for v in self.read(self.pipelines[pipeline][1], "broad_phase_shape_pairs", row)[0])
        if self.contacts is None:
            return ()
        names = ("rigid_contact_shape0", "rigid_contact_shape1") if kind == "contact" else ("soft_contact_shape",)
        if kind not in ("contact", "soft_contact"):
            return ()
        return tuple(int(self.read(self.contacts, name, index)[0]) for name in names)

    def world_for(self, key: TreeKey) -> int:
        """Resolve a contact or pair through its shapes, then soft-contact particles, with -1 for shared data."""
        for shape in self.shapes(key):
            if 0 <= shape < self.workspace.model.shape_count:
                world = int(self.read(self.workspace.model, "shape_world", shape)[0])
                if world >= 0:
                    return world
        if key[0] == "soft_contact":
            indices = self.read(self.contacts, "soft_contact_indices", key[1])[0]
            for particle in indices:
                if 0 <= particle < self.workspace.model.particle_count:
                    return int(self.read(self.workspace.model, "particle_world", int(particle))[0])
        return -1

    def points(self, key: TreeKey | None) -> list[np.ndarray]:
        """Transfer only the selected witness rows and their owning transforms."""
        if key is None or key[0] not in ("contact", "soft_contact") or self.state is None:
            return []
        ws, index, result = self.workspace, key[1], []
        shapes = self.shapes(key)
        for side, shape in enumerate(shapes):
            if shape < 0 or shape >= ws.model.shape_count:
                continue
            body = int(self.read(ws.model, "shape_body", shape)[0])
            xform = wp.transform_identity()
            if body >= 0:
                pose = self.read(self.state, "body_q", body)[0]
                xform = wp.transform(wp.vec3(pose[:3]), wp.quat(pose[3:]))
            if key[0] == "contact":
                point = self.read(self.contacts, f"rigid_contact_point{side}", index)[0]
                point = point + self.read(self.contacts, f"rigid_contact_offset{side}", index)[0]
            else:
                point = self.read(self.contacts, "soft_contact_body_pos", index)[0]
            point = wp.transform_point(xform, wp.vec3(point))
            world = int(self.read(ws.model, "shape_world", shape)[0])
            if world < 0:
                world = self.world_for(key)
            if world >= 0 and not ws.viewer._should_render_world(world):
                continue
            if world >= 0 and ws.viewer.world_offsets is not None:
                point += wp.vec3(self.read(ws.viewer, "world_offsets", world)[0])
            result.append(np.asarray(wp.transform_point(ws.viewer.layer.xform, point)))
        return result

    def label_points(self, rows: tuple[TreeKey, ...]) -> dict[TreeKey, np.ndarray]:
        """Gather selected witnesses once per paused snapshot, with one rigid readback."""
        ws = self.workspace
        key = ("label_points", rows, tuple(ws.viewer.layer.xform), tuple(ws.viewer._visible_worlds or ()))
        if key not in self.cache:
            rigid = [row for kind, row in rows if kind == "contact"]
            points = {}
            if rigid:
                indices = wp.array(rigid, dtype=int, device=ws.model.device)
                output = wp.empty((len(rigid), 2), dtype=wp.vec3, device=ws.model.device)
                contacts = self.contacts
                wp.launch(
                    _rigid_label_points,
                    dim=(len(rigid), 2),
                    inputs=[
                        indices,
                        self.state.body_q,
                        ws.model.shape_body,
                        ws.model.shape_world,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
                        ws.viewer.world_offsets,
                        ws.viewer._visible_worlds_mask,
                        ws.viewer.layer.xform,
                    ],
                    outputs=[output],
                    device=ws.model.device,
                )
                points.update((("contact", row), value) for row, value in zip(rigid, output.numpy(), strict=True))
            points.update((row, np.asarray(self.points(row))) for row in rows if row[0] == "soft_contact")
            self.cache[key] = points
        return self.cache[key]

    def _field(
        self,
        inspector: PropertyInspector,
        name: str,
        value: Any,
        spec: Model.AttributeSpec | None = None,
        *,
        start: int = 0,
        count: int = 1,
    ) -> None:
        """Render a read-only contact attribute slice with metadata-driven types and reference navigation."""
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        ui.push_id(name)
        if isinstance(value, wp.array):
            if start + count > len(value):
                w.text_value(name, "N/A")
                ui.pop_id()
                return
            cache_key = id(value), start, count
            if cache_key not in self.cache:
                self.cache[cache_key] = value[start : start + count].numpy()
            values = self.cache[cache_key]
        else:
            values = np.asarray([value])
        spec = spec or Model.AttributeSpec(Frequency.ONCE)
        with w.property_tooltip(lambda: attribute_summary(name, spec, value, Assignment.CONTACT)):
            if spec.references is not None and values.dtype.kind in "iu":
                inspector._references(
                    name.rsplit(":", maxsplit=1)[-1],
                    values,
                    spec,
                    AttributeRows(ws.model, spec.frequency, [0]),
                    start,
                    "Collision output; read only.",
                )
            else:
                inspector._typed(
                    name.rsplit(":", maxsplit=1)[-1],
                    values,
                    getattr(value, "dtype", None),
                    False,
                    "Collision output; read only.",
                    name,
                )
        ui.pop_id()

    def contact_fields(self, key: TreeKey) -> Iterator[tuple[str | None, str, Any, Model.AttributeSpec, int, int]]:
        """Yield exact row slices, honoring registered frequencies and row widths."""
        kind, index = key
        rigid = kind == "contact"
        frequency = Frequency.CONTACT_RIGID if rigid else Frequency.CONTACT_SOFT
        prefix = "rigid_contact_" if rigid else "soft_contact_"
        references = {
            "shape0": Frequency.SHAPE,
            "shape1": Frequency.SHAPE,
            "shape": Frequency.SHAPE,
            "particle": Frequency.PARTICLE,
            "indices": Frequency.PARTICLE,
        }
        for name, value in vars(self.contacts).items():
            if name.startswith(prefix) and isinstance(value, wp.array) and len(value) > index:
                if not self.workspace._at_live_frame():
                    timeline = getattr(self.workspace.example, "timeline", self.workspace.history)
                    if timeline is None or name not in timeline.frames[timeline.index].contacts:
                        continue
                suffix = name[len(prefix) :]
                if suffix in ("count", "tids") or suffix.startswith("diff_"):
                    continue
                yield None, name, value, Model.AttributeSpec(frequency, references=references.get(suffix)), index, 1
        observables = getattr(self.workspace.viewer, "_last_contact_observables", None)
        if observables is not None:
            for name, field in observables._observable_fields().values():
                value = getattr(observables, name, None)
                if value is None or field.frequency not in (frequency, Frequency.CONTACT):
                    continue
                row = index + (
                    self.contacts.rigid_contact_max if not rigid and field.frequency == Frequency.CONTACT else 0
                )
                yield "Solver observables", name, value, Model.AttributeSpec(field.frequency), row, 1
        if not self.workspace._at_live_frame():
            return
        for name, spec in self.workspace.model._iter_attribute_specs():
            if spec.assignment != Assignment.CONTACT or spec.deprecated:
                continue
            _, _, value = resolve_attribute(self.contacts, name)
            if value is None:
                continue
            row = index
            if spec.frequency == Frequency.CONTACT:
                row += 0 if rigid else self.contacts.rigid_contact_max
            elif spec.frequency == Frequency.ONCE:
                row = 0
            elif spec.frequency != frequency:
                continue
            namespace = name.split(":")[0] if ":" in name else "Global"
            yield namespace, name, value, spec, row * spec.row_width, spec.row_width

    def draw(self, inspector: PropertyInspector) -> None:
        """Draw the selected collision resource, refreshing its paused snapshot when requested."""
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        self.sync()
        key = ws.selected
        if w.button("Refresh", small=True):
            self.sync(force=True)
            key = ws.selected
        ui.same_line()
        w.label("Paused frame" if ws.viewer.is_paused() else "Pause to inspect contacts", muted=True)
        if not ws._at_live_frame():
            ui.text_wrapped(
                "Recorded contacts. Pipeline buffers and unrecorded attributes are available only at the live frame."
            )
        if key[0] in ("contact", "soft_contact", "collision_pair"):
            if ui.begin_tab_bar("##collision_contact_properties"):
                if w.tab_item("Contacts")[0]:
                    self.contacts_tab = True
                    if key[0] == "collision_pair":
                        for side, shape in enumerate(self.shapes(key)):
                            self._field(
                                inspector,
                                f"Shape {side}",
                                shape,
                                Model.AttributeSpec(Frequency.ONCE, references=Frequency.SHAPE),
                            )
                        ui.separator_text("Contacts for this pair")
                        self.draw_contact_list(inspector, key)
                    elif self.contacts is not None:
                        self.draw_contact(inspector, key)
                    ui.end_tab_item()
                ui.end_tab_bar()
            return
        if key[0] in ("collision_pipeline", "collision_broad", "collision_narrow"):
            pipeline = self.pipelines[key[1]][1]
            owner = {
                "collision_pipeline": pipeline,
                "collision_broad": pipeline.broad_phase,
                "collision_narrow": pipeline.narrow_phase,
            }[key[0]]
            ui.text_disabled("Settings · read only")
            ui.text_wrapped(
                "Buffers describe the last collision pass; inspecting them does not run collision detection."
            )
            if getattr(getattr(ws.example, "solver", None), "use_mujoco_contacts", False):
                ui.text_wrapped("The solver uses internal contacts. This Newton pipeline may not have been executed.")
            bvhs = ws.tree.resources.bvhs
            if bvhs.root in ws.tree.labels and w.button("Inspect BVHs", small=True):
                ws.navigate(bvhs.root)
                return
            for name, value in vars(owner).items():
                if not name.startswith("_") and isinstance(value, (str, int, float, bool, Enum, np.generic)):
                    self._field(inspector, name, value.value if isinstance(value, Enum) else value)
            if not ws.viewer.is_paused() or not ws._at_live_frame():
                return
            with w.group("Last collision pass") as opened:
                if opened:
                    counters = pipeline if key[0] == "collision_broad" else owner
                    for name, array in vars(counters).items():
                        if (
                            not name.startswith("_")
                            and name.endswith("_count")
                            and isinstance(array, wp.array)
                            and len(array) == 1
                        ):
                            self._field(inspector, name, array)
            with w.group("Buffers", default_open=False) as opened:
                if opened:
                    for name, array in vars(owner).items():
                        if name.startswith("_") or not isinstance(array, wp.array) or not len(array):
                            continue
                        with w.disclosure(f"{name} · {array.shape}", default_open=False) as show:
                            if show:
                                ui.push_id(name)
                                for row in inspector._page(range(len(array))):
                                    self._field(inspector, f"{name}[{row}]", array, start=row)
                                ui.pop_id()
            return
        if not ws.viewer.is_paused():
            return
        if self.contacts is None:
            ui.text_wrapped("This example has not supplied a Contacts buffer to the viewer.")
            return
        for label, count, raw, cap in zip(
            ("Rigid contacts", "Soft-rigid contacts"),
            self.counts,
            self.raw_counts,
            (self.contacts.rigid_contact_max, self.contacts.soft_contact_max),
            strict=True,
        ):
            w.text_value(label, f"{count:,} / {cap:,}")
            if raw > cap:
                ui.text_wrapped(f"{label}: buffer overflow ({raw:,} requested).")
        if not self.pipelines:
            ui.text_wrapped("No Newton CollisionPipeline found on this example, model, or solver.")

    def draw_contact(self, inspector: PropertyInspector, key: TreeKey) -> None:
        """Draw contact geometry, matching candidate pairs, and built-in or namespaced custom attributes."""
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        points = self.points(key)
        with w.group("Geometry") as opened:
            if opened:
                for side, point in enumerate(points):
                    w.vector(f"Point {side} · display", point)
                if len(points) == 2:
                    normal = self.read(self.contacts, "rigid_contact_normal", key[1])[0]
                    normal = np.asarray(wp.transform_vector(ws.viewer.layer.xform, wp.vec3(normal)))
                    w.number("Separation", float(np.dot(points[1] - points[0], normal)), unit="m")
        if key[0] == "contact" and ws._at_live_frame() and self.pipelines and ws.selected[0] != "collision_pair":
            with w.group("Candidate pairs") as opened:
                if opened:
                    pairs = self.candidate_pairs(key)
                    if not pairs:
                        ui.text_wrapped("No matching pair in the available broadphase buffers.")
                    for pair in inspector._page(pairs):
                        pipeline, index = pair[1]
                        name = self.pipelines[pipeline][0].replace("_", " ")
                        ui.push_id(f"pair_{pipeline}_{index}")
                        if w.button(f"Pair {index} · {name}"):
                            self.reveal(pair)
                            ws.navigate(pair)
                        ui.pop_id()
        groups = defaultdict(list)
        for namespace, *field in self.contact_fields(key):
            groups[namespace].append(field)
        builtin_fields = groups.pop(None, ())
        with w.group("Contacts") as opened:
            if opened:
                for field in builtin_fields:
                    name, value, spec, start, count = field
                    self._field(inspector, name, value, spec, start=start, count=count)
        if groups:
            ui.separator_text("Custom attributes")
            if ui.begin_tab_bar("##contact_namespaces"):
                for namespace, fields in groups.items():
                    if w.tab_item(namespace)[0]:
                        for name, value, spec, start, count in fields:
                            self._field(inspector, name, value, spec, start=start, count=count)
                        ui.end_tab_item()
                ui.end_tab_bar()

    def shape_contacts(self, shape: int) -> tuple[TreeKey, ...]:
        """Filter active rows once per paused frame; never transfer buffer capacity."""
        return self.object_contacts(("shape", shape))

    def object_contacts(self, key: TreeKey) -> tuple[TreeKey, ...]:
        """Match shape ownership or an unordered candidate pair against active contacts."""
        if not self.workspace.viewer.is_paused() or self.contacts is None:
            return ()
        cache_key = ("object_contacts", key)
        if cache_key not in self.cache:
            rigid, soft = self.counts
            a = self.read(self.contacts, "rigid_contact_shape0", count=rigid)
            b = self.read(self.contacts, "rigid_contact_shape1", count=rigid)
            c = self.read(self.contacts, "soft_contact_shape", count=soft)
            kind, index = key
            if kind == "collision_pair":
                first, second = self.shapes(key)
                rigid_mask = ((a == first) & (b == second)) | ((a == second) & (b == first))
                soft_mask = np.zeros(soft, dtype=bool)
            else:
                shapes = (index,) if kind == "shape" else self.workspace.model.body_shapes.get(index, ())
                rigid_mask = np.isin(a, shapes) | np.isin(b, shapes)
                soft_mask = np.isin(c, shapes)
            self.cache[cache_key] = tuple(("contact", int(i)) for i in np.flatnonzero(rigid_mask)) + tuple(
                ("soft_contact", int(i)) for i in np.flatnonzero(soft_mask)
            )
        return self.cache[cache_key]

    def candidate_pairs(self, key: TreeKey) -> tuple[TreeKey, ...]:
        """Link a rigid contact to matching rows of the last available broadphase pass."""
        if not self.workspace.viewer.is_paused() or not self.workspace._at_live_frame() or key[0] != "contact":
            return ()
        shapes = self.shapes(key)
        if len(shapes) != 2:
            return ()
        first, second = sorted(shapes)
        cache_key = ("candidate_pairs", first, second)
        if cache_key not in self.cache:
            result = []
            for i, (_, pipeline) in enumerate(self.pipelines):
                group = self.groups.get(("collision_pairs", i))
                count = group[2] if group else 0
                if not count:
                    continue
                pairs = self.read(pipeline, "broad_phase_shape_pairs", count=count)
                mask = ((pairs[:, 0] == first) & (pairs[:, 1] == second)) | (
                    (pairs[:, 0] == second) & (pairs[:, 1] == first)
                )
                result.extend(("collision_pair", (i, int(row))) for row in np.flatnonzero(mask))
            self.cache[cache_key] = tuple(result)
        return self.cache[cache_key]

    def number(self, key: TreeKey) -> int:
        """Return a one-based display number, placing soft contacts after the rigid-contact range."""
        return key[1] + 1 + (self.counts[0] if key[0] == "soft_contact" else 0)

    def contact_label(self, key: TreeKey) -> str:
        """Name both witnesses, retaining parent names when leaves coincide."""
        model = self.workspace.model
        shapes = self.shapes(key)
        paths = [
            str(model.shape_label[index]).rstrip("/").split("/")
            if 0 <= index < model.shape_count and model.shape_label[index]
            else [f"Shape {index}" if index >= 0 else "N/A"]
            for index in shapes
        ]
        labels = [path[-1] for path in paths]
        if len(labels) == 2 and labels[0] == labels[1]:
            for depth in range(2, max(map(len, paths)) + 1):
                labels = ["/".join(path[-depth:]) for path in paths]
                if labels[0] != labels[1]:
                    break
            if labels[0] == labels[1] and shapes[0] != shapes[1]:
                labels = [f"{label} [{index}]" for label, index in zip(labels, shapes, strict=True)]
        if key[0] == "soft_contact":
            indices = self.read(self.contacts, "soft_contact_indices", key[1])[0]
            particles = ", ".join(str(int(index)) for index in indices if index >= 0)
            labels.insert(0, f"Particle {particles}" if particles else "N/A")
        return f"#{self.number(key)} · " + " ↔ ".join(labels)

    def draw_object(self, inspector: PropertyInspector, owner: TreeKey) -> None:
        """Open an object contact view, showing live annotations or an inspectable paused contact list."""
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        self.contacts_tab = True
        self.sync()
        if not ws.viewer.is_paused():
            self.list_owner = owner
            owner_key = (self.token, owner)
            if getattr(self, "_owner_key", None) != owner_key:
                ws.viewer.show_contacts = True
                self._owner_key = owner_key
            ui.text_wrapped("Pause to show contact labels and inspect contacts.")
            return
        if w.button("Refresh", small=True):
            self.sync(force=True)
        ui.same_line()
        self.draw_contact_list(inspector, owner)

    def draw_contact_list(self, inspector: PropertyInspector, owner: TreeKey) -> None:
        """Paginate matching contacts and draw the active contact properties.

        Pagination limits inspector rows only; viewport annotation still covers
        every matching contact. Changing the list owner clears the active row.
        """
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        self.contacts_tab = True
        rows = self.object_contacts(owner)
        w.label(f"{len(rows):,} active contacts", muted=True)
        if owner != self.list_owner:
            self.active_contact = None
            self.list_owner = owner
        if not rows:
            self.active_contact = None
            self.visible_contacts = ()
            return
        if self.active_contact not in rows:
            self.active_contact = None
        # Paginate the inspector only. Every matching contact remains annotated.
        self.visible_contacts = tuple(inspector._page(rows))
        owner_key = (self.token, owner)
        if getattr(self, "_owner_key", None) != owner_key:
            ws.viewer.show_contacts = True
            self._owner_key = owner_key
        for key in self.visible_contacts:
            ui.push_id(f"{key[0]}_{key[1]}")
            label = self.contact_label(key)
            if ui.selectable(label + "##contact", self.active_contact == key)[0]:
                self.active_contact = key
            ui.pop_id()
        if self.active_contact is None:
            return
        ui.separator_text(f"Contact #{self.number(self.active_contact)}")
        if not ws._at_live_frame():
            ui.text_wrapped("Recorded contact. Unrecorded custom attributes are available only at the live frame.")
        self.draw_contact(inspector, self.active_contact)

    def overlays(self) -> None:
        """Synchronize renderer contact labels and selection rings with the current inspector context.

        Clear annotations during playback or when the context is hidden. Keep
        selection-ring updates independent of label layout and cache paused
        labels until their contact or display inputs change.
        """
        ws, viewer = self.workspace, self.workspace.viewer
        renderer = getattr(viewer, "renderer", None)
        if renderer is None:
            return
        direct = ws.selected and ws.selected[0] in ("contact", "soft_contact")
        listed = self.contacts_tab and self.list_owner == ws.selected
        if not viewer.is_paused() or self.tree is None or not viewer.show_contacts or not (direct or listed):
            # Contact indices are transient slots; live labels flicker and
            # require host readbacks without providing stable identities.
            renderer._contact_labels = ()
            renderer._contact_selection = None
            self._overlay_key = None
            return
        self.sync()
        selected = ws.selected if direct else self.active_contact if listed else None
        sides = 2 if viewer.show_contact_pairs else 1
        renderer._contact_selection = (
            (viewer._qualify("/contacts/modes"), selected[1] * sides, sides)
            if selected is not None and selected[0] == "contact"
            else None
        )
        # Selecting geometry must not copy witnesses or run text layout. Keep
        # the inexpensive selected-contact ring independent of label visibility.
        if not self.contacts_tab or not viewer.show_contact_labels:
            renderer._contact_labels = ()
            self._overlay_key = None
            return
        rows = (ws.selected,) if direct else self.object_contacts(ws.selected)
        key = (
            self.token,
            ws.selected,
            rows,
            selected,
            tuple(viewer.layer.xform),
            tuple(viewer._visible_worlds or ()),
            viewer.show_contact_labels,
            viewer.show_contact_pairs,
        )
        if key == self._overlay_key:
            return
        self._overlay_key = key
        renderer._contact_label_scale = ws.widgets.scale
        # The GL pass anchors these labels in scene depth before translucent
        # surfaces are composited, just like the contact glyphs themselves.
        points = self.label_points(rows)
        renderer._contact_labels = tuple(
            (points[row][:sides], str(self.number(row)), row == selected) for row in rows if row in points
        )
