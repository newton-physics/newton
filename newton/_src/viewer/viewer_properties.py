# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metadata-driven, bounded property inspection for the local workspace."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from enum import IntEnum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp

from ..core.types import Axis
from ..geometry.types import GeoType
from ..sim.enums import JointTargetMode, JointType
from ..sim.model import Model
from ..sim.state import State
from .viewer_attributes import (
    Assignment,
    Frequency,
    PropertyDomain,
    attribute_summary,
    component_attributes,
    frequency_row_label,
    resolve_attribute,
)
from .viewer_frequencies import FrequencyCatalog
from .viewer_widgets import PropertyEdit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from newton import Control
    from newton._src.viewer.viewer_types import TreeKey
    from newton._src.viewer.viewer_workspace import ViewerWorkspace


def joint_editor_options(
    attr: str, lower: float | None = None, upper: float | None = None
) -> dict[str, float | bool | tuple[float, float] | None]:
    """Keep simulation limits while using a human-sized drag for very wide ranges."""
    speed = 0.001 if attr in ("joint_q", "joint_target_q") else 0.01 if attr in ("joint_qd", "joint_target_qd") else 0.1
    bounds = (
        (float(lower), float(upper))
        if lower is not None and upper is not None and np.isfinite([lower, upper]).all() and lower < upper
        else None
    )
    # A slider covers its entire range in one gesture. Very wide and sentinel
    # limits belong in a constant-speed drag field, with Ctrl-click text entry.
    slider = bounds is not None and bounds[1] - bounds[0] <= 100.0
    return {"speed": speed, "bounds": bounds, "slider": slider}


class PropertyInspector:
    """Map registered State, Control, and Model attributes to typed GUI fields.

    Selection domains are cached by Model identity and selection. Attribute
    availability is resolved from current component storage, so optional and
    namespaced fields need no solver-specific UI. Only open groups and visible
    pages request values through the workspace's bounded read cache.

    Widgets propose edits; the workspace authorizes and applies them to live
    storage. Model fields and reference indices remain read-only because changing
    them may invalidate topology or solver allocations.
    """

    kinds: ClassVar[dict[str, Model.AttributeFrequency]] = {
        "joint": Frequency.JOINT,
        "body": Frequency.BODY,
        "shape": Frequency.SHAPE,
        "articulation": Frequency.ARTICULATION,
        "particle": Frequency.PARTICLE,
        "tri": Frequency.TRIANGLE,
        "tet": Frequency.TETRAHEDRON,
        "edge": Frequency.EDGE,
        "spring": Frequency.SPRING,
    }

    def __init__(self, workspace: ViewerWorkspace) -> None:
        self.workspace = workspace
        self.context_key = None
        self.domains: list[PropertyDomain] = []
        self.component = Assignment.STATE

    def clear(self) -> None:
        """Release model domains while preserving the selected component tab."""
        self.context_key = None
        self.domains.clear()

    def _contexts(self) -> None:
        """Merge selected row domains without merging different Models or frequencies."""
        ws = self.workspace
        selection = frozenset(getattr(ws, "selection", ()) or (ws.selected or ("scene", 0),))
        key = (id(ws.model), selection)
        if self.context_key == key:
            return
        self.context_key = key
        domains = [
            domain
            for selected in sorted(selection, key=lambda item: (item[0], repr(item[1])))
            for domain in self._selection_domains(selected)
        ]
        if len(selection) == 1:
            # Preserve lazy ranges for whole worlds and large deformables.
            self.domains = domains
            return
        merged: dict[tuple[int, Model.AttributeFrequency | str, bool], list[PropertyDomain]] = defaultdict(list)
        for domain in domains:
            key = (id(domain.model), domain.frequency, domain.joints)
            merged[key].append(domain)
        self.domains = [
            PropertyDomain(
                group[0].model,
                group[0].frequency,
                sorted({int(index) for domain in group for index in domain.indices}),
                joints=group[0].joints,
            )
            for group in merged.values()
        ]

    def _selection_domains(self, selected: TreeKey) -> list[PropertyDomain]:
        """Resolve one object into registered row domains, including related vertices and DOFs."""
        ws = self.workspace
        domains: list[PropertyDomain] = []
        model = ws.model
        kind, index = selected

        def add(
            frequency: Model.AttributeFrequency | str,
            indices: Sequence[int] | np.ndarray,
            *,
            source: Model = model,
            joints: bool = False,
        ) -> None:
            if joints and len(indices):
                first, last = int(indices[0]), int(indices[-1])
                offsets = "joint_q_start" if frequency == Frequency.JOINT_COORD else "joint_qd_start"
                spans = ws.read(source, offsets, first, last - first + 2)
                indices = np.asarray(indices, dtype=int)
                indices = indices[spans[indices - first + 1] > spans[indices - first]]
            if len(indices):
                domains.append(PropertyDomain(source, frequency, indices, joints=joints))

        if kind in self.kinds:
            if kind == "particle":
                source = ws.tree.deformables.source(index)
                add(Frequency.PARTICLE, [index - source.offset], source=source.model)
            else:
                add(self.kinds[kind], [index])
                if kind in ("tri", "tet"):
                    particles, _, _ = ws.tree.deformables.elements(selected)
                    add(Frequency.PARTICLE, particles)
            if kind in ("joint", "articulation"):
                joints = [index]
                if kind == "articulation":
                    start = int(ws.read(model, "articulation_start", index)[0])
                    end = int(ws.read(model, "articulation_end", index)[0])
                    owners = ws.read(model, "joint_articulation", start, end - start)
                    joints = np.flatnonzero(owners == index) + start
                    add(Frequency.JOINT, joints)
                    if model.custom_frequency_articulation:
                        if ws.tree is None:
                            ws._ensure_tree()
                        for frequency, rows in ws.tree.custom.articulations.get(index, {}).items():
                            add(frequency, rows)
                add(Frequency.JOINT_COORD, joints, joints=True)
                add(Frequency.JOINT_DOF, joints, joints=True)
        elif (frequency := FrequencyCatalog.frequency((kind, index))) is not None:
            add(frequency, [index])
        elif kind == "frequency":
            frequency, indices = ws.tree.custom.groups[selected]
            add(frequency, indices)
        elif kind.startswith("attribute:"):
            add(Frequency[kind.split(":", 1)[1]], [index])
        elif kind == "reference_group":
            frequency = ws.tree.meta[selected]["frequency"]
            add(frequency, [child[1] for child in ws.tree.children[selected]])
        elif kind in ("surface", "volume", "particle_set", "elements"):
            particles, triangles, tets = ws.tree.deformables.elements(selected)
            for source in ws.tree.deformables.sources:
                if isinstance(particles, range) and particles.step == 1:
                    local = range(
                        max(particles.start, source.offset) - source.offset,
                        min(particles.stop, source.end) - source.offset,
                    )
                else:
                    indices = np.asarray(particles)
                    local = indices[(indices >= source.offset) & (indices < source.end)] - source.offset
                add(Frequency.PARTICLE, local, source=source.model)
            add(Frequency.TRIANGLE, triangles)
            add(Frequency.TETRAHEDRON, tets)
        elif kind in ("scene", "root", "group"):
            add(Frequency.ONCE, [0])
            if kind == "root" and index >= 0:
                add(Frequency.WORLD, [index])
            else:
                add(Frequency.WORLD, range(model.world_count))
            for frequency, count in model.custom_frequency_counts.items():
                add(frequency, range(count))
        return domains

    def _owner(self, domain: PropertyDomain, component: Model.AttributeAssignment) -> Model | State | Control | None:
        ws = self.workspace
        if component == Assignment.MODEL:
            return domain.model
        if component == Assignment.CONTROL:
            if domain.model is not ws.model:
                return None
            if ws.history is not None and ws.history.replaying:
                return ws.history.display_control
            return ws.session.control
        if domain.model is ws.model:
            return getattr(ws.viewer, "_last_state", None) or ws.live_state()
        return next((s.state for s in ws.tree.deformables.sources if s.model is domain.model), None)

    def draw(self) -> None:
        ws, ui, w = self.workspace, self.workspace.gui.ui.imgui, self.workspace.widgets
        self._contexts()
        components = {
            component: self._collect(component)
            for component in (Assignment.STATE, Assignment.CONTROL, Assignment.MODEL)
        }
        available = [
            component
            for component, groups in components.items()
            if groups or (component == Assignment.MODEL and self._counts())
        ]
        contact_owner = (
            len(ws.selection) == 1
            and ws.selected is not None
            and ws.selected[0] in ("shape", "body")
            and ws.collision.root in ws.tree.labels
        )
        ws.collision.contacts_tab = False
        if contact_owner:
            available.append("contacts")
        fallback = available[0] if available and self.component not in available else None
        if ui.begin_tab_bar("##component_properties"):
            for title, component in (
                ("State", Assignment.STATE),
                ("Control", Assignment.CONTROL),
                ("Model", Assignment.MODEL),
            ):
                if component not in available:
                    continue
                if w.tab_item(title, flags=ui.TabItemFlags_.set_selected if component == fallback else 0)[0]:
                    self.component = component
                    self._component(component, components[component])
                    ui.end_tab_item()
            if contact_owner and w.tab_item("Contacts")[0]:
                self.component = "contacts"
                ws.collision.draw_object(self, ws.selected)
                ui.end_tab_item()
            ui.end_tab_bar()
        if ws._undo is not None:
            owner, attr, start, values, joint = ws._undo
            editable = ws.can_edit_control() if ws.is_control_owner(owner) else ws.can_edit_state()
            ui.begin_disabled(not editable)
            if w.button("Undo last edit"):
                if isinstance(owner, State) and attr in ("joint_q", "joint_qd"):
                    owner = ws.live_state()
                if ws.apply_edit(owner, attr, start, values, joint_index=joint, save_undo=False):
                    ws._undo = None
            ui.end_disabled()

    def _counts(self) -> dict[str, int]:
        ws = self.workspace
        if ws.selected and ws.selected[0] not in ("root", "group"):
            return {}
        return {
            name: getattr(ws.model, name, 0)
            for name in dict.fromkeys(Model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS.values())
            if getattr(ws.model, name, 0)
        }

    def _collect(
        self, component: Model.AttributeAssignment
    ) -> dict[str | None, list[tuple[PropertyDomain, str, Model.AttributeSpec, object, str, Any]]]:
        """Determine populated groups from metadata and storage, without reading live arrays."""
        groups = defaultdict(list)
        attributes = {}
        for domain in self.domains:
            owner = self._owner(domain, component)
            if owner is None:
                continue
            key = (id(domain.model), id(owner))
            if key not in attributes:
                attributes[key] = defaultdict(list)
                for field in component_attributes(domain.model, owner, component):
                    attributes[key][field[1].frequency].append(field)
            for name, spec, parent, attr, value in attributes[key].get(domain.frequency, ()):
                if isinstance(value, (wp.array, np.ndarray, list, tuple)):
                    first = int(domain.indices[0])
                    if domain.joints:
                        offsets = "joint_q_start" if domain.frequency == Frequency.JOINT_COORD else "joint_qd_start"
                        span = self.workspace.read(domain.model, offsets, first, 2)
                        stop = int(span[1])
                    else:
                        stop = (first + 1) * spec.row_width
                    if stop > len(value):
                        continue
                custom = spec.assignment is not None or ":" in name or name not in domain.model._CORE_ATTRIBUTE_SPECS
                # None identifies built-in fields; the empty namespace is a
                # distinct tab for unnamespaced custom attributes.
                namespace = (name.split(":", 1)[0] if ":" in name else "") if custom else None
                groups[namespace].append((domain, name, spec, parent, attr, value))
        return groups

    def _component(
        self,
        component: Model.AttributeAssignment,
        groups: dict[str | None, list[tuple[PropertyDomain, str, Model.AttributeSpec, object, str, Any]]] | None = None,
    ) -> None:
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        groups = self._collect(component) if groups is None else groups
        single_shape = len(ws.selection) == 1 and ws.selected and ws.selected[0] == "shape"
        if component == Assignment.MODEL and single_shape:
            ws.tree.resources.shape_summary(ws, ws.selected[1])
        if component == Assignment.CONTROL:
            editable = ws.can_edit_control()
            if w.icon_button(
                "##reset_control",
                "reset",
                "Restore all Control inputs to the example's initial values."
                if editable
                else "Return to the live frame to change Control inputs.",
                label="Reset Control",
                size=22,
                enabled=editable,
            ):
                ws.reset_control()
            ui.same_line(spacing=6 * w.scale)
            if w.icon_button(
                "##zero_control",
                "clear",
                "Zero all Control forces, velocities, activations and custom inputs; preserve initial position targets."
                if editable
                else "Return to the live frame to change Control inputs.",
                label="Zero inputs",
                size=22,
                enabled=editable,
            ):
                ws.reset_control(zero=True)
            ui.separator()
        if component == Assignment.MODEL and (counts := self._counts()):
            with w.group("Counts") as opened:
                if opened:
                    for name, count in counts.items():
                        w.number(name.removesuffix("_count").replace("_", " ").title(), count, integer=True)
        ui.push_id(int(component))
        for namespace, fields in groups.items():
            if namespace is not None:
                continue
            if component == Assignment.MODEL:
                domains = defaultdict(list)
                for field in fields:
                    domains[field[0].frequency].append(field)
                for frequency, entries in domains.items():
                    title = frequency.name.replace("_", " ").title() if isinstance(frequency, Frequency) else frequency
                    with w.group(
                        title, default_open=frequency not in (Frequency.JOINT_COORD, Frequency.JOINT_DOF)
                    ) as opened:
                        if opened:
                            self._fields(entries, component)
            else:
                self._fields(fields, component)
        custom = [name for name in groups if name is not None]
        if custom:
            ui.separator_text("Custom attributes")
            if ui.begin_tab_bar("##custom_namespaces", ui.TabBarFlags_.fitting_policy_scroll):
                for namespace in custom:
                    if w.tab_item(f"{namespace or 'Global'}##namespace:{namespace}")[0]:
                        self._fields(groups[namespace], component, flat=True)
                        ui.end_tab_item()
                ui.end_tab_bar()
        if component == Assignment.MODEL and single_shape:
            ws._shape_solver_properties(ws.selected[1])
        ui.pop_id()

    def _fields(
        self,
        fields: Sequence[tuple[PropertyDomain, str, Model.AttributeSpec, object, str, Any]],
        component: Model.AttributeAssignment,
        *,
        flat: bool = False,
    ) -> None:
        w, ui = self.workspace.widgets, self.workspace.gui.ui.imgui
        # Attribute groups keep large articulations and meshes bounded and useful:
        # q for all joints together, then qd for all joints, with stable row labels.
        for domain, name, spec, owner, attr, array in fields:
            ui.push_id(f"{id(domain.model)}:{name}")
            with w.property_tooltip(
                lambda name=name, spec=spec, array=array: attribute_summary(name, spec, array, component)
            ):
                count = len(domain.indices)
                if count > 1:
                    group = w.disclosure if flat else w.group
                    with group(name.split(":")[-1], default_open=component != Assignment.MODEL) as opened:
                        if opened:
                            visible = self._page(domain.indices)
                            self._prefetch(domain, spec, owner, attr, array, visible, component)
                            for index in visible:
                                self._field(domain, name, spec, owner, attr, array, int(index), component, nested=True)
                else:
                    self._field(domain, name, spec, owner, attr, array, int(domain.indices[0]), component)
            ui.pop_id()

    def _prefetch(
        self,
        domain: PropertyDomain,
        spec: Model.AttributeSpec,
        owner: object,
        attr: str,
        array: Any,
        indices: Sequence[int] | np.ndarray,
        component: Model.AttributeAssignment,
    ) -> None:
        """Read a compact visible page once instead of synchronizing once per field."""
        if not isinstance(array, wp.array) or len(indices) < 2:
            return
        ws = self.workspace
        first, last = int(indices[0]), int(indices[-1])
        # Sparse selections must not pull large intervening regions from other worlds.
        if last - first > 4 * len(indices):
            return
        source = (
            ws.joint_display_state() if component == Assignment.STATE and attr in ("joint_q", "joint_qd") else owner
        )
        if domain.joints:
            offsets = "joint_q_start" if domain.frequency == Frequency.JOINT_COORD else "joint_qd_start"
            spans = ws.read(domain.model, offsets, first, last - first + 2).astype(int)
            rows = [(int(i), int(spans[i - first]), int(spans[i - first + 1] - spans[i - first])) for i in indices]
            for index, start, count in rows:
                ws._cache[(id(domain.model), offsets, index, 2)] = np.array([start, start + count])
        else:
            rows = [(int(i), int(i) * spec.row_width, spec.row_width) for i in indices]
        start, stop = rows[0][1], rows[-1][1] + rows[-1][2]
        if stop <= start or stop > len(array):
            return
        live = component != Assignment.MODEL
        values = ws.read(source, attr, start, stop - start, live=live)
        cache = ws._live_cache if live else ws._cache
        for _, offset, count in rows:
            cache[(id(source), attr, offset, count)] = values[offset - start : offset - start + count]

    def _page(self, indices: Sequence[int] | np.ndarray) -> Sequence[int] | np.ndarray:
        ui, w = self.workspace.gui.ui.imgui, self.workspace.widgets
        size = 16
        if len(indices) <= size:
            return indices
        storage, key = ui.get_state_storage(), ui.get_id("##rows")
        pages = (len(indices) + size - 1) // size
        page = min(storage.get_int(key), pages - 1)
        if w.icon_button("##prev", "previous", "Previous rows", enabled=page > 0, size=22):
            page -= 1
        ui.same_line()
        if w.icon_button("##next", "play", "Next rows", enabled=page + 1 < pages, size=22):
            page += 1
        ui.same_line()
        edit = w.number("##page", page + 1, editable=True, integer=True, bounds=(1, pages), width=55 * w.scale)
        if edit.changed:
            page = edit.value - 1
        storage.set_int(key, page)
        ui.same_line()
        w.label(f"/ {pages} · {len(indices):,} rows", muted=True)
        return indices[page * size : (page + 1) * size]

    def _field(
        self,
        domain: PropertyDomain,
        name: str,
        spec: Model.AttributeSpec,
        owner: object,
        attr: str,
        array: Any,
        index: int,
        component: Model.AttributeAssignment,
        *,
        nested: bool = False,
    ) -> None:
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        model = domain.model
        start, count = index * spec.row_width, spec.row_width
        joint = index if domain.joints else None
        if domain.joints:
            # A joint is one explorer item but owns a variable-length span in
            # coordinate/DOF storage. row_width alone cannot describe that span.
            offsets = "joint_q_start" if domain.frequency == Frequency.JOINT_COORD else "joint_qd_start"
            span = ws.read(model, offsets, index, 2).astype(int)
            start, count = int(span[0]), int(span[1] - span[0])
        if count <= 0:
            return
        label = attr
        if nested:
            if domain.joints or domain.frequency == Frequency.JOINT:
                label = model.joint_label[index].rstrip("/").rsplit("/", 1)[-1] or f"Joint {index}"
            else:
                label = frequency_row_label(domain.frequency, index)
                label_attribute = model.custom_frequency_label_attributes.get(domain.frequency)
                if label_attribute:
                    _, _, labels = resolve_attribute(model, label_attribute)
                    if labels is not None and index < len(labels):
                        label = str(labels[index]).rstrip("/").rsplit("/", 1)[-1] or label
        ui.push_id(index)
        if isinstance(array, IntEnum):
            if isinstance(array, Axis):
                w.axis(label, array)
            else:
                w.enum(label, array, list(type(array)))
            ui.pop_id()
            return
        if isinstance(array, wp.array):
            if start < 0 or start + count > len(array):
                ui.pop_id()
                return
            source = (
                ws.joint_display_state() if component == Assignment.STATE and attr in ("joint_q", "joint_qd") else owner
            )
            values = ws.read(source, attr, start, count, live=component != Assignment.MODEL)
        elif isinstance(array, (list, tuple, np.ndarray)):
            values = np.asarray(array[start : start + count])
        else:
            values = np.asarray([array])
        editable = component == Assignment.CONTROL or (
            component == Assignment.STATE and (joint is not None or spec.assignment == Assignment.STATE)
        )
        allowed = ws.can_edit_control() if component == Assignment.CONTROL else ws.can_edit_state()
        editable &= spec.references is None and isinstance(array, wp.array) and allowed
        reason = (
            "Return to the live frame to edit Control inputs."
            if component == Assignment.CONTROL
            else "Pause at the live frame to edit."
        )
        if component == Assignment.MODEL:
            reason = ""
        elif component == Assignment.STATE and joint is None and spec.assignment != Assignment.STATE:
            reason = "Solver state. Edit joint coordinates to change articulated body motion."
        if spec.references is not None and component != Assignment.MODEL:
            reason = "Reference indices are read only."
        if joint is not None:
            kind = JointType(int(ws.read(model, "joint_type", joint)[0]))
            art = int(ws.read(model, "joint_articulation", joint)[0])
            if component == Assignment.STATE and (kind == JointType.ROD or art < 0):
                editable = False
                reason = "This joint is advanced by the solver, outside forward kinematics."
        if spec.references is not None:
            self._references(label, values, spec, domain, start, reason)
            edit = PropertyEdit(values)
        elif joint is not None and name in ("joint_q", "joint_qd", "joint_target_q", "joint_target_qd", "joint_f"):
            edit = self._joint_widget(label, values, model, joint, kind, attr, editable, reason)
        else:
            edit = self._typed(
                label,
                values,
                getattr(array, "dtype", None),
                editable,
                reason,
                name,
                frequency=domain.frequency if domain.joints else None,
                start=start,
            )
        if editable:
            live_owner = ws.live_state() if component == Assignment.STATE and joint is not None else owner
            if edit.activated:
                ws._drag_undo = (live_owner, attr, start, values.copy(), joint)
            if edit.changed:
                if ws.apply_edit(live_owner, attr, start, edit.value, joint_index=joint, save_undo=False):
                    ws._undo = ws._drag_undo or (live_owner, attr, start, values.copy(), joint)
            if edit.finished:
                ws._drag_undo = None
        ui.pop_id()

    def reference_target(self, model: Model, frequency: Model.AttributeFrequency | str, index: int) -> TreeKey | None:
        """Resolve an index to an explorer selection without changing selection."""
        ws = self.workspace
        if frequency in (Frequency.JOINT_COORD, Frequency.JOINT_DOF):
            return None
        count = model._attribute_frequency_count(frequency)
        if index < 0 or index >= count:
            return None
        if frequency == Frequency.PARTICLE:
            source = next((s for s in ws.tree.deformables.sources if s.model is model), None)
            return ("particle", source.offset + index) if source is not None else None
        if model is not ws.model:
            return None
        if isinstance(frequency, str):
            return FrequencyCatalog.key(frequency, index)
        if frequency == Frequency.WORLD:
            key = ("root", index)
            return key if key in ws.tree.roots else ("root", -1)
        for kind, domain in self.kinds.items():
            if frequency == domain and kind not in ("edge", "spring"):
                return kind, index
        return "attribute:" + frequency.name, index

    def reference_label(
        self, model: Model, frequency: Model.AttributeFrequency | str, index: int, target: TreeKey
    ) -> str:
        """Read host labels without materializing the referenced world or page."""
        tree = self.workspace.tree
        if isinstance(frequency, str):
            return tree.custom.label(frequency, index)
        kind = target[0] if target[0] in self.kinds else frequency.name.lower()
        labels = getattr(model, kind + "_label", ())
        if index < len(labels) and isinstance(labels[index], str) and labels[index].strip():
            return labels[index]
        return tree.labels.get(target, frequency_row_label(frequency, index))

    def _references(
        self, label: str, values: ArrayLike, spec: Model.AttributeSpec, domain: PropertyDomain, start: int, reason: str
    ) -> None:
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        values = np.asarray(values)
        flat = values.reshape(-1)

        def row(label: str, value: int) -> None:
            if spec.references in (Frequency.JOINT_COORD, Frequency.JOINT_DOF):
                w.number(label, int(value), integer=True, reason=reason)
                return
            target = self.reference_target(domain.model, spec.references, int(value))
            # Exclusive ends describe ranges, not an object at the next index.
            if spec.compaction_policy == "end":
                target = None
            title = None
            if target is not None:
                title = self.reference_label(domain.model, spec.references, int(value), target)
            caption = ws.tree.reference_caption(target, title) if target is not None else None
            if w.reference(label, int(value), target=title, caption=caption, reason=reason):
                ws.navigate(target)

        if flat.size == 1:
            row(label, flat[0])
        else:
            with w.compound(label, reason=reason):
                for index in self._page(range(flat.size)):
                    ui.push_id(index)
                    frequency = domain.frequency if domain.joints and values.ndim == 1 else None
                    row(frequency_row_label(frequency, start + index if frequency is not None else index), flat[index])
                    ui.pop_id()

    def _typed(
        self,
        label: str,
        values: ArrayLike,
        dtype: Any,
        editable: bool,
        reason: str,
        attr: str,
        *,
        frequency: Model.AttributeFrequency | str | None = None,
        start: int = 0,
    ) -> PropertyEdit:
        w, ui = self.workspace.widgets, self.workspace.gui.ui.imgui
        values = np.asarray(values)
        args = {"editable": editable, "reason": reason}
        flat = values.reshape(-1)
        if len(values) > 1 and (
            frequency is not None
            or getattr(dtype, "_shape_", ())
            or dtype == wp.bool
            or values.dtype.kind in "bOUS"
            or values.dtype.names
        ):
            result = PropertyEdit(values.copy())
            ui.push_id(label)
            with w.compound(label, **args):
                for index in self._page(range(len(values))):
                    ui.push_id(index)
                    edit = self._typed(
                        frequency_row_label(frequency, start + index),
                        values[index : index + 1],
                        dtype,
                        editable,
                        reason,
                        attr,
                    )
                    if edit.changed:
                        result.value[index] = np.asarray(edit.value).reshape(np.shape(values[index]))
                    result.changed |= edit.changed
                    result.activated |= edit.activated
                    result.finished |= edit.finished
                    ui.pop_id()
            ui.pop_id()
            return result
        if values.dtype.names:
            ui.push_id(label)
            with w.compound(label, reason=reason):
                for field in values.dtype.names:
                    self._typed(field, values[field], None, False, "", field)
            ui.pop_id()
            return PropertyEdit(values)
        if values.dtype.kind in "OUS":
            if flat.size == 1:
                w.text_value(label, flat[0], reason=reason)
            else:
                ui.push_id(label)
                with w.compound(label, reason=reason):
                    for index in self._page(range(flat.size)):
                        w.text_value(frequency_row_label(None, index), flat[index], reason=reason)
                ui.pop_id()
            return PropertyEdit(values)
        if (dtype == wp.bool or values.dtype.kind == "b") and flat.size == 1:
            return w.boolean(label, bool(flat[0]), **args)
        enums = {"shape_type": GeoType, "joint_type": JointType, "joint_target_mode": JointTargetMode}
        if attr in enums and flat.size == 1:
            return w.enum(label, int(flat[0]), list(enums[attr]), **args)
        if dtype == wp.transform and len(values) == 1:
            return w.transform(label, flat, **args)
        if dtype == wp.quat and len(values) == 1:
            return w.quat(label, flat, **args)
        if dtype == wp.spatial_vector and len(values) == 1:
            return w.spatial_vector(label, flat, force=attr.endswith("_f"), **args)
        if dtype == wp.mat33 and len(values) == 1:
            return w.matrix3(label, flat, **args)
        integer = values.dtype.kind in "iu"
        if integer and (np.any(flat > np.iinfo(np.int32).max) or np.any(flat < np.iinfo(np.int32).min)):
            w._label(label, False, reason="Integer outside the 32-bit editor range.", inline=True)
            ui.text_wrapped(", ".join(str(int(v)) for v in flat))
            return PropertyEdit(values)
        if flat.size == 1:
            return w.number(label, flat[0], integer=integer, **args)
        if not integer and getattr(dtype, "_length_", 0) in (2, 3, 4) and len(values) == 1:
            return w.vector(label, flat, **args)
        shape = values.shape[1:]
        labels = (
            [f"{r}{c}" for r in range(shape[0]) for c in range(shape[1])]
            if len(shape) == 2 and len(values) == 1
            else None
        )
        return w.array(label, flat, integer=integer, labels=labels, **args)

    def _joint_widget(
        self,
        label: str,
        values: ArrayLike,
        model: Model,
        joint: int,
        kind: JointType,
        attr: str,
        editable: bool,
        reason: str,
    ) -> PropertyEdit:
        ws, w = self.workspace, self.workspace.widgets
        values = values.reshape(-1)
        coord = attr in ("joint_q", "joint_target_q")
        args = {"editable": editable, "reason": reason}
        if coord and kind in (JointType.FREE, JointType.DISTANCE) and len(values) == 7:
            return w.transform(label, values, **args)
        if coord and kind == JointType.BALL and len(values) == 4:
            return w.quat(label, values, **args)
        if len(values) == 6 and kind in (JointType.FREE, JointType.DISTANCE):
            return w.spatial_vector(label, values, force=attr == "joint_f", position=attr == "joint_target_q", **args)
        dof_start = int(ws.read(model, "joint_qd_start", joint)[0])
        linear = int(ws.read(model, "joint_dof_dim", joint)[0, 0])
        limits = coord and kind in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6)
        lower = ws.read(model, "joint_limit_lower", dof_start, len(values)) if limits else []
        upper = ws.read(model, "joint_limit_upper", dof_start, len(values)) if limits else []
        if attr in ("joint_qd", "joint_target_qd"):
            upper = ws.read(model, "joint_velocity_limit", dof_start, len(values))
            lower = -upper
            limits = len(upper) == len(values)
        result = PropertyEdit(values.copy())
        with w.compound(label, **args) if len(values) > 1 else nullcontext():
            for i, value in enumerate(values):
                editor = joint_editor_options(attr, lower[i], upper[i]) if limits else joint_editor_options(attr)
                unit = (
                    ("m" if i < linear else "rad")
                    if attr in ("joint_q", "joint_target_q")
                    else ("m/s" if i < linear else "rad/s")
                )
                if attr == "joint_f":
                    unit = "N" if i < linear else "N m"
                component = (
                    (f"Linear {i + 1}" if i < linear else f"Angular {i - linear + 1}") if len(values) > 1 else label
                )
                edit = w.number(component, value, unit=unit, **editor, **args)
                result.value[i] = edit.value
                result.changed |= edit.changed
                result.activated |= edit.activated
                result.finished |= edit.finished
        return result
