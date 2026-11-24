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

from ...core.types import Axis
from ...geometry.types import GeoType
from ...sim.enums import JointTargetMode, JointType
from ...sim.model import Model
from ...sim.state import State
from ...sim.tendon import TendonGuideType
from ...solvers.solver import SolverBase
from ..viewer_attributes import (
    Assignment,
    AttributeRows,
    Frequency,
    attribute_summary,
    component_attributes,
    frequency_row_label,
    resolve_attribute,
)
from ..viewer_frequencies import CustomFrequencyIndex
from .widgets import PropertyEdit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from newton import Control
    from newton._src.viewer.gui.workspace import ViewerWorkspace
    from newton._src.viewer.viewer_types import TreeKey


def joint_editor_options(
    attr: str, lower: float | None = None, upper: float | None = None
) -> dict[str, float | bool | tuple[float, float] | None]:
    """Keep simulation limits while using a human-sized drag for very wide ranges."""
    speed = (
        0.001
        if attr in ("joint_q", "joint_target_q", "joint_rest_q")
        else 0.01
        if attr in ("joint_qd", "joint_target_qd")
        else 0.1
    )
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
        "curve": Frequency.CURVE,
        "tendon": Frequency.TENDON,
        "tendon_guide": Frequency.TENDON_GUIDE,
        "tendon_segment": Frequency.TENDON_SEGMENT,
        "attachment_particle_body": Frequency.ATTACHMENT_PARTICLE_BODY,
        "constraint_mimic": Frequency.CONSTRAINT_MIMIC,
        "actuator": Frequency.ACTUATOR,
    }

    def __init__(self, workspace: ViewerWorkspace) -> None:
        """Initialize selection-derived property domains and the active component tab."""
        self.workspace = workspace
        self.context_key = None
        self.domains: list[AttributeRows] = []
        self.component = Assignment.STATE
        self._spec_cache: dict[
            Model,
            tuple[tuple[dict[str, Any], ...], dict[Frequency | str, list[tuple[str, Model.AttributeSpec]]]],
        ] = {}

    def clear(self) -> None:
        """Release model domains while preserving the selected component tab."""
        self.context_key = None
        self.domains.clear()
        self._spec_cache.clear()

    def _specs(self, model: Model) -> dict[Frequency | str, list[tuple[str, Model.AttributeSpec]]]:
        """Index immutable specs by frequency, retaining no State/Control storage.

        Registry maps remain mutable for late solver registration and legacy
        frequency/assignment overrides. Compare shallow snapshots so those
        changes invalidate immediately without rebuilding unified metadata for
        every tab on every frame. AttributeSpec itself is frozen.
        """
        mappings = (model.attribute_specs, model.attribute_frequency, model.attribute_assignment)
        cached = self._spec_cache.get(model)
        if cached is None or cached[0] != mappings:
            indexed = defaultdict(list)
            for name, spec in model._iter_attribute_specs():
                indexed[spec.frequency].append((name, spec))
            cached = tuple(mapping.copy() for mapping in mappings), dict(indexed)
            self._spec_cache[model] = cached
        return cached[1]

    def _contexts(self) -> None:
        """Merge selected row domains without merging different Models or frequencies."""
        ws = self.workspace
        selection = frozenset(getattr(ws, "selection", ()) or (ws.selected or ("scene", 0),))
        key = (id(ws.model), selection)
        if self.context_key == key:
            return
        if self.context_key is not None and self.context_key[0] != key[0]:
            self._spec_cache.clear()
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
        merged: dict[tuple[int, Model.AttributeFrequency | str, bool], list[AttributeRows]] = defaultdict(list)
        for domain in domains:
            key = (id(domain.model), domain.frequency, domain.joints)
            merged[key].append(domain)
        self.domains = [
            AttributeRows(
                group[0].model,
                group[0].frequency,
                sorted({int(index) for domain in group for index in domain.indices}),
                joints=group[0].joints,
            )
            for group in merged.values()
        ]

    def _selection_domains(self, selected: TreeKey) -> list[AttributeRows]:
        """Resolve one object into registered row domains, including related vertices and DOFs."""
        ws = self.workspace
        domains: list[AttributeRows] = []
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
                domains.append(AttributeRows(source, frequency, indices, joints=joints))

        if kind in self.kinds:
            if kind == "particle":
                source = ws.tree.deformables.source(index)
                add(Frequency.PARTICLE, [index - source.offset], source=source.model)
            else:
                add(self.kinds[kind], [index])
                if kind in ("tri", "tet", "edge"):
                    particles, _, _ = ws.tree.deformables.elements(selected)
                    add(Frequency.PARTICLE, particles)
                elif kind == "curve":
                    bodies, joints = ws.tree.deformables.curve_elements(index)
                    add(Frequency.BODY, bodies)
                    add(Frequency.JOINT, joints)
                    add(Frequency.JOINT_COORD, joints, joints=True)
                    add(Frequency.JOINT_DOF, joints, joints=True)
                elif kind == "tendon":
                    guides, segments = ws.tree.tendon_elements(index)
                    add(Frequency.TENDON_GUIDE, guides)
                    add(Frequency.TENDON_SEGMENT, segments)
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
        elif (frequency := CustomFrequencyIndex.frequency((kind, index))) is not None:
            add(frequency, [index])
        elif kind == "frequency":
            frequency, indices = ws.tree.custom.groups[selected]
            add(frequency, indices)
        elif kind.startswith("attribute:"):
            add(Frequency[kind.split(":", 1)[1]], [index])
        elif kind == "reference_group":
            frequency = ws.tree.meta[selected]["frequency"]
            add(frequency, [child[1] for child in ws.tree.children[selected]])
        elif kind == "tendon_group":
            add(Frequency.TENDON, ws.tree.tendons_in_world(index))
        elif kind == "attachment_group":
            add(Frequency.ATTACHMENT_PARTICLE_BODY, ws.tree.attachments_in_world(index))
        elif kind == "attachment_page":
            add(Frequency.ATTACHMENT_PARTICLE_BODY, ws.tree.meta[selected].get("indices", ()))
        elif kind == "mimic_group":
            add(Frequency.CONSTRAINT_MIMIC, ws.tree.mimics_in_world(index))
        elif kind == "actuator_group":
            if 0 <= index < len(model.actuators):
                act_indices = model.actuators[index].actuator_indices
                if act_indices is not None and len(act_indices):
                    add(Frequency.ACTUATOR, act_indices.numpy().tolist())
        elif kind in ("surface", "volume", "particle_set", "elements"):
            if kind in ("surface", "volume"):
                obj = ws.tree.deformables.objects[index]
                if obj.model_index is not None:
                    add(Frequency.SURFACE if kind == "surface" else Frequency.VOLUME, [obj.model_index])
            particles, triangles, tets = ws.tree.deformables.elements(selected)
            edges = ws.tree.deformables.element_edges(selected)
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
            add(Frequency.EDGE, edges)
            add(Frequency.TETRAHEDRON, tets)
        elif kind in ("scene", "root", "group"):
            add(Frequency.ONCE, [0])
            if kind == "root":
                add(Frequency.WORLD, [index if index >= 0 else model.world_count])
            elif kind == "group":
                world = ws.tree.world_for(selected)
                add(Frequency.WORLD, [world if world >= 0 else model.world_count])
            else:
                add(Frequency.WORLD, range(model.world_count + 1))
            for frequency, count in model.custom_frequency_counts.items():
                add(frequency, range(count))
        return domains

    def _owner(self, domain: AttributeRows, component: Model.AttributeAssignment) -> Model | State | Control | None:
        """Resolve the Model, displayed State, or Control for a property domain.

        Historical Control comes from the display snapshot; secondary particle
        Models can supply their own logged State but do not inherit the primary
        Model Control. Return None when the component is unavailable.
        """
        ws = self.workspace
        if component == Assignment.MODEL:
            return domain.model
        if component == Assignment.CONTROL:
            if domain.model is not ws.model:
                return None
            if ws.history is not None and ws.history.replaying:
                return ws.history.display_control
            return ws.bindings.control
        if domain.model is ws.model:
            return getattr(ws.viewer, "_last_state", None) or ws.live_state()
        return next((s.state for s in ws.tree.deformables.sources if s.model is domain.model), None)

    def draw(self) -> None:
        """Draw nonempty component tabs for the selection and offer undo when the recorded edit is writable."""
        ws, ui, w = self.workspace, self.workspace.gui.ui.imgui, self.workspace.widgets
        self._contexts()
        components = {
            component: self._collect(component)
            for component in (Assignment.STATE, Assignment.CONTROL, Assignment.MODEL)
        }
        observed = self._observables()
        observed_2d = self._observables_2d()
        summary_only = (
            len(ws.selection) == 1
            and ws.selected is not None
            and ws.selected[0] in ("deformable_visual_mesh", "actuator_collection", "actuator_group", "battery")
        )
        available = [
            component
            for component, groups in components.items()
            if groups
            or (component == Assignment.MODEL and (self._counts() or summary_only))
            or (component == Assignment.STATE and (observed or observed_2d))
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
                    if component == Assignment.STATE and (observed or observed_2d):
                        with w.group("Solver observables") as opened:
                            if opened:
                                self._fields(observed, component)
                                self._draw_observables_2d(observed_2d)
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

    def _observables(self) -> list[tuple[AttributeRows, str, Model.AttributeSpec, object, str, Any]]:
        """Match active solver observables to the selected non-contact row domains.

        Contact observables belong to the Contacts tab. One-dimensional arrays
        whose rows follow a selected frequency (including joint DOF/coordinate
        spans on selected joints and articulations) are listed here.
        """
        ws = self.workspace
        logged_obs = getattr(ws, "logged_observables", None)
        observables = (
            logged_obs()
            if callable(logged_obs)
            else (
                getattr(ws.viewer, "_last_observables", None) or getattr(ws.viewer, "_last_contact_observables", None)
            )
        )
        if observables is None or observables.model is not ws.model:
            return []
        contact = (Frequency.CONTACT, Frequency.CONTACT_RIGID, Frequency.CONTACT_SOFT)
        joint_freqs = (Frequency.JOINT_COORD, Frequency.JOINT_DOF)
        result = []
        for kind, (name, field) in observables._observable_fields().items():
            if kind not in observables.kinds:
                continue
            value = getattr(observables, name, None)
            if field.frequency in contact or not isinstance(value, wp.array) or value.ndim != 1:
                continue
            spec = Model.AttributeSpec(field.frequency)
            for domain in self.domains:
                if domain.model is ws.model and domain.frequency == field.frequency:
                    if domain.joints and field.frequency not in joint_freqs:
                        continue
                    result.append((domain, name, spec, observables, name, value))
        return result

    def _observables_2d(self) -> list[tuple[AttributeRows, str, Model.AttributeSpec, wp.array]]:
        """Match 2D ``(steps, world_count + 1)`` per-world solver observables to selected worlds."""
        ws = self.workspace
        logged_obs = getattr(ws, "logged_observables", None)
        observables = (
            logged_obs()
            if callable(logged_obs)
            else (
                getattr(ws.viewer, "_last_observables", None) or getattr(ws.viewer, "_last_contact_observables", None)
            )
        )
        if observables is None or observables.model is not ws.model:
            return []
        result = []
        for kind, (name, field) in observables._observable_fields().items():
            if kind not in observables.kinds or field.frequency != Frequency.WORLD:
                continue
            value = getattr(observables, name, None)
            if not isinstance(value, wp.array) or value.ndim != 2 or value.size == 0:
                continue
            spec = Model.AttributeSpec(field.frequency)
            for domain in self.domains:
                if domain.model is ws.model and domain.frequency == Frequency.WORLD and not domain.joints:
                    result.append((domain, name, spec, value))
        return result

    def _draw_observables_2d(self, entries: Sequence[tuple[AttributeRows, str, Model.AttributeSpec, wp.array]]) -> None:
        """Render 2D per-world observables (such as per-iteration residual history) for selected worlds."""
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        reason = "Computed by the solver during step(); read-only."
        for domain, name, spec, array in entries:
            ui.push_id(f"{id(domain.model)}:2d:{name}")
            with w.property_tooltip(
                lambda name=name, spec=spec, array=array: attribute_summary(name, spec, array, Assignment.STATE)
            ):
                indices = [int(i) for i in domain.indices if 0 <= int(i) < array.shape[1]]
                if len(indices) == 1:
                    w.array(name, ws._read_world_column(array, indices[0]), reason=reason)
                elif len(indices) > 1:
                    with w.group(name, default_open=True) as opened:
                        if opened:
                            for slot in self._page(indices):
                                label = (
                                    "shared world (-1)"
                                    if slot == domain.model.world_count
                                    else frequency_row_label(Frequency.WORLD, slot)
                                )
                                ui.push_id(int(slot))
                                w.array(label, ws._read_world_column(array, int(slot)), reason=reason)
                                ui.pop_id()
            ui.pop_id()

    def _counts(self) -> dict[str, int]:
        """Return nonzero Model frequency counts for root or group selection; otherwise return an empty mapping."""
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
    ) -> dict[str | None, list[tuple[AttributeRows, str, Model.AttributeSpec, object, str, Any]]]:
        """Determine populated groups from metadata and storage, without reading live arrays."""
        groups = defaultdict(list)
        attributes = {}
        frequencies = defaultdict(set)
        for domain in self.domains:
            frequencies[id(domain.model)].add(domain.frequency)
        for domain in self.domains:
            owner = self._owner(domain, component)
            if owner is None:
                continue
            key = (id(domain.model), id(owner))
            if key not in attributes:
                attributes[key] = defaultdict(list)
                specs = self._specs(domain.model)
                for field in component_attributes(
                    domain.model,
                    owner,
                    component,
                    specs=(entry for frequency in frequencies[id(domain.model)] for entry in specs.get(frequency, ())),
                ):
                    attributes[key][field[1].frequency].append(field)
            for name, spec, parent, attr, value in attributes[key].get(domain.frequency, ()):
                if isinstance(value, (wp.array, np.ndarray, list, tuple)):
                    first = int(domain.indices[0])
                    if domain.joints:
                        offsets = "joint_q_start" if domain.frequency == Frequency.JOINT_COORD else "joint_qd_start"
                        span = self.workspace.read(domain.model, offsets, first, 2)
                        stop = int(span[1]) * spec.row_width
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
        groups: dict[str | None, list[tuple[AttributeRows, str, Model.AttributeSpec, object, str, Any]]] | None = None,
    ) -> None:
        """Draw one component with built-in fields, namespace tabs, and applicable resource or Control actions."""
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        groups = self._collect(component) if groups is None else groups
        single = ws.selected if len(ws.selection) == 1 else None
        if component == Assignment.MODEL and single is not None:
            if single[0] == "shape":
                ws.tree.resources.shape_summary(ws, single[1])
            elif single[0] == "deformable_visual_mesh":
                ws.tree.deformables.visual_mesh_summary(ws, single[1])
            elif single[0] == "actuator_collection":
                with w.group("Actuators & Batteries") as opened:
                    if opened:
                        w.number("Actuator groups", len(ws.model.actuators), integer=True)
                        w.number("Actuator count", ws.model.actuator_count, integer=True)
                        w.number("Battery groups", len(ws.model.batteries), integer=True)
            elif single[0] == "actuator_group":
                self._actuator_summary(single[1])
            elif single[0] == "battery":
                self._battery_summary(single[1])
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
        ui.pop_id()

    def _fields(
        self,
        fields: Sequence[tuple[AttributeRows, str, Model.AttributeSpec, object, str, Any]],
        component: Model.AttributeAssignment,
        *,
        flat: bool = False,
        default_open: bool | None = None,
    ) -> None:
        """Render attributes in domain order, paginating selected objects within each attribute.

        Prefetch only visible rows and use stable Model, attribute, and row IDs.
        Flat groups avoid adding another panel around custom attributes.
        """
        w, ui = self.workspace.widgets, self.workspace.gui.ui.imgui
        is_open = (component != Assignment.MODEL) if default_open is None else default_open
        # Attribute groups keep large articulations and meshes bounded and useful:
        # q for all joints together, then qd for all joints, with stable row labels.
        for domain, name, spec, owner, attr, array in fields:
            ui.push_id(f"{id(domain.model)}:{name}:{int(domain.joints)}")
            with w.property_tooltip(
                lambda name=name, spec=spec, array=array: attribute_summary(name, spec, array, component)
            ):
                count = len(domain.indices)
                if count > 1:
                    group = w.disclosure if flat else w.group
                    with group(name.split(":")[-1], default_open=is_open) as opened:
                        if opened:
                            visible = self._page(domain.indices)
                            self._prefetch(domain, name, spec, owner, attr, array, visible, component)
                            for index in visible:
                                self._field(domain, name, spec, owner, attr, array, int(index), component, nested=True)
                else:
                    self._field(domain, name, spec, owner, attr, array, int(domain.indices[0]), component)
            ui.pop_id()

    def _prefetch(
        self,
        domain: AttributeRows,
        name: str,
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
        joint_state = component == Assignment.STATE and name in ("joint_q", "joint_qd")
        source = ws.joint_display_state() if joint_state else owner
        if domain.joints:
            offsets = "joint_q_start" if domain.frequency == Frequency.JOINT_COORD else "joint_qd_start"
            spans = ws.read(domain.model, offsets, first, last - first + 2).astype(int)
            rows = [(int(i), int(spans[i - first]), int(spans[i - first + 1] - spans[i - first])) for i in indices]
            for index, start, count in rows:
                ws._cache[(id(domain.model), offsets, index, 2)] = np.array([start, start + count])
            rows = [(index, start * spec.row_width, count * spec.row_width) for index, start, count in rows]
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
        """Draw pagination when needed and return at most 16 rows, retaining the page in the current ImGui ID scope."""
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

    def _actuator_summary(self, index: int) -> None:
        """Draw read-only composition and attribute bindings for one Model Actuator group."""
        ws, w = self.workspace, self.workspace.widgets
        actuators = ws.model.actuators
        if not (0 <= index < len(actuators)):
            return
        act = actuators[index]
        with w.group("Actuator") as opened:
            if opened:
                w.text_value("Drive", type(act.drive).__name__)
                w.number("Actuators", act.num_actuators, integer=True)
                if act.motor is not None:
                    w.text_value("Motor", type(act.motor).__name__)
                    w.boolean("Uses battery", bool(act.motor.uses_battery))
                if act.input_processors:
                    w.text_value("Input processors", ", ".join(type(proc).__name__ for proc in act.input_processors))
                if act.clamping:
                    w.text_value("Clamping", ", ".join(type(clamp).__name__ for clamp in act.clamping))
        with w.group("Bindings", default_open=False) as opened:
            if opened:
                for label, attr in (
                    ("State position", "state_pos_attr"),
                    ("State velocity", "state_vel_attr"),
                    ("Target position", "control_target_pos_attr"),
                    ("Target velocity", "control_target_vel_attr"),
                    ("Feedforward", "control_feedforward_attr"),
                    ("Output", "control_output_attr"),
                    ("Computed output", "control_computed_output_attr"),
                ):
                    value = getattr(act, attr, None)
                    if value is not None:
                        w.text_value(label, str(value))

    def _battery_summary(self, index: int) -> None:
        """Draw read-only parameters for one Model Battery group."""
        ws, w = self.workspace, self.workspace.widgets
        batteries = ws.model.batteries
        if not (0 <= index < len(batteries)):
            return
        bat = batteries[index]
        with w.group("Battery") as opened:
            if opened:
                w.text_value("Type", type(bat).__name__)
                w.number("Batteries", bat.num_batteries, integer=True)
                for attr, label, unit in (
                    ("nominal_voltage", "Nominal voltage", "V"),
                    ("resistance", "Internal resistance", "Ohm"),
                    ("min_voltage", "Min voltage", "V"),
                ):
                    array = getattr(bat, attr, None)
                    if isinstance(array, wp.array) and len(array):
                        values = ws.read(bat, attr, 0, min(len(array), 16))
                        if len(values) == 1:
                            w.number(label, float(values[0]), unit=unit)
                        else:
                            w.array(label, values)

    def _field(
        self,
        domain: AttributeRows,
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
        """Resolve one frequency row, render its editor, and apply permitted edits with drag-level undo.

        Joint coordinate and DOF domains use joint offset spans rather than
        fixed row widths. Writes target the live owner only after workspace
        editability checks; displayed historical data remains read-only.
        """
        ws, w, ui = self.workspace, self.workspace.widgets, self.workspace.gui.ui.imgui
        model = domain.model
        start, count = index * spec.row_width, spec.row_width
        joint = index if domain.joints else None
        # Only built-in reduced coordinates participate in inverse/forward
        # kinematics. A custom namespace can legitimately reuse their names.
        joint_state = component == Assignment.STATE and name in ("joint_q", "joint_qd")
        if domain.joints:
            # A joint is one explorer item but owns a variable-length span in
            # coordinate/DOF storage. row_width alone cannot describe that span.
            offsets = "joint_q_start" if domain.frequency == Frequency.JOINT_COORD else "joint_qd_start"
            span = ws.read(model, offsets, index, 2).astype(int)
            start, count = int(span[0]) * spec.row_width, int(span[1] - span[0]) * spec.row_width
        if count <= 0:
            return
        label = attr
        if nested:
            if domain.joints or domain.frequency == Frequency.JOINT:
                label = model.joint_label[index].rstrip("/").rsplit("/", 1)[-1] or f"Joint {index}"
            elif domain.frequency == Frequency.WORLD and index == model.world_count:
                label = "shared world (-1)"
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
            source = ws.joint_display_state() if joint_state else owner
            values = ws.read(source, attr, start, count, live=component != Assignment.MODEL)
        elif isinstance(array, (list, tuple, np.ndarray)):
            values = np.asarray(array[start : start + count])
        else:
            values = np.asarray([array])
        editable = component == Assignment.CONTROL or (
            component == Assignment.STATE and (joint_state or spec.assignment == Assignment.STATE)
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
        if isinstance(owner, SolverBase.Observables):
            editable = False
            reason = "Computed by the solver during step(); read-only."
        if joint is not None:
            kind = JointType(int(ws.read(model, "joint_type", joint)[0]))
            art = int(ws.read(model, "joint_articulation", joint)[0])
            if joint_state and (kind == JointType.ROD or art < 0):
                editable = False
                reason = "This joint is advanced by the solver, outside forward kinematics."
        if spec.references is not None:
            self._references(label, values, spec, domain, start, reason)
            edit = PropertyEdit(values)
        elif (
            component == Assignment.MODEL
            and name == "body_variant"
            and len(values) == 1
            and (variant_labels := model.body_variant_labels(start))
        ):
            can_switch = ws.can_edit_state()
            edit = w.choice(
                label,
                int(values[0]),
                {i: text or f"Variant {i}" for i, text in enumerate(variant_labels)},
                editable=can_switch,
                reason="Pause at the live frame to switch body variants.",
            )
            if can_switch and edit.changed:
                ws.select_body_variant(start, int(edit.value))
            ui.pop_id()
            return
        elif joint is not None and name in (
            "joint_q",
            "joint_qd",
            "joint_target_q",
            "joint_target_qd",
            "joint_rest_q",
            "joint_f",
            "joint_act",
        ):
            edit = self._joint_widget(label, values, model, joint, kind, attr, editable, reason)
        else:
            # Joint offsets count logical coordinates/DOFs, whereas row_width
            # counts flattened entries per row. Preserve that distinction in
            # both the storage slice and the nested row labels.
            edit = self._typed(
                label,
                values,
                getattr(array, "dtype", None),
                editable,
                reason,
                name,
                frequency=domain.frequency if domain.joints else None,
                start=start // spec.row_width if domain.joints else start,
                row_width=spec.row_width if domain.joints else 1,
            )
        if editable:
            live_owner = ws.live_state() if joint_state else owner
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
        if frequency == Frequency.WORLD:
            if model is not ws.model:
                return None
            if index in (-1, model.world_count):
                return ("root", -1) if ("root", -1) in ws.tree.roots else None
            if 0 <= index < model.world_count:
                key = ("root", index)
                return key if key in ws.tree.roots else ("root", -1) if ("root", -1) in ws.tree.roots else None
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
            return CustomFrequencyIndex.key(frequency, index)
        if frequency in (Frequency.SURFACE, Frequency.VOLUME):
            target = ws.tree.deformables.object_key_for_model(frequency.name.lower(), index, ws.tree)
            if target is not None:
                return target
        for kind, domain in self.kinds.items():
            if frequency == domain and kind not in ("spring", "actuator"):
                return kind, index
        return "attribute:" + frequency.name, index

    def reference_label(
        self, model: Model, frequency: Model.AttributeFrequency | str, index: int, target: TreeKey
    ) -> str:
        """Read host labels without materializing the referenced world or page."""
        tree = self.workspace.tree
        if isinstance(frequency, str):
            return tree.custom.label(frequency, index)
        if frequency == Frequency.WORLD and index in (-1, model.world_count):
            return "Shared (-1)"
        kind = target[0] if target[0] in self.kinds else frequency.name.lower()
        labels = getattr(model, kind + "_label", ())
        if index < len(labels) and isinstance(labels[index], str) and labels[index].strip():
            return labels[index]
        return tree.labels.get(target, frequency_row_label(frequency, index))

    def _references(
        self, label: str, values: ArrayLike, spec: Model.AttributeSpec, domain: AttributeRows, start: int, reason: str
    ) -> None:
        """Draw integer references with navigation to valid target objects.

        Coordinate and DOF indices, missing targets, and exclusive range ends
        have no navigation action. Compound references retain frequency labels.
        """
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
        elif domain.joints and spec.row_width > 1:
            with w.compound(label, reason=reason):
                for index in self._page(range(len(values) // spec.row_width)):
                    ui.push_id(index)
                    with w.compound(
                        frequency_row_label(domain.frequency, start // spec.row_width + index), reason=reason
                    ):
                        entries = values[index * spec.row_width : (index + 1) * spec.row_width].reshape(-1)
                        for component in self._page(range(len(entries))):
                            row(str(component), entries[component])
                    ui.pop_id()
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
        row_width: int = 1,
    ) -> PropertyEdit:
        """Choose widgets from array shape and dtype and return an unapplied PropertyEdit.

        Preserve compound row structure (including flattened ``row_width``
        entries per logical frequency row), paginate repeated values, and use
        specialized physics widgets where the dtype identifies their meaning.
        The caller owns writing changed values back to the source array.
        """
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
                for index in self._page(range(len(values) // row_width)):
                    ui.push_id(index)
                    first, end = index * row_width, (index + 1) * row_width
                    edit = self._typed(
                        frequency_row_label(frequency, start + index),
                        values[first:end],
                        dtype,
                        editable,
                        reason,
                        attr,
                    )
                    if edit.changed:
                        result.value[first:end] = np.asarray(edit.value).reshape(values[first:end].shape)
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
        enums = {
            "shape_type": GeoType,
            "joint_type": JointType,
            "joint_target_mode": JointTargetMode,
            "tendon_guide_type": TendonGuideType,
        }
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
        """Render a joint span with units, representations, and limits appropriate to its type.

        Use transform, quaternion, or spatial-vector editors when applicable;
        otherwise distinguish linear and angular components. Finite limits
        enable bounded sliders; unbounded values use controlled drag speeds.
        """
        ws, w = self.workspace, self.workspace.widgets
        values = values.reshape(-1)
        coord = attr in ("joint_q", "joint_target_q", "joint_rest_q")
        args = {"editable": editable, "reason": reason}
        if coord and kind in (JointType.FREE, JointType.DISTANCE, JointType.ROD) and len(values) == 7:
            return w.transform(label, values, **args)
        if coord and kind == JointType.BALL and len(values) == 4:
            return w.quat(label, values, **args)
        if len(values) == 6 and kind in (JointType.FREE, JointType.DISTANCE, JointType.ROD):
            return w.spatial_vector(
                label,
                values,
                force=attr in ("joint_f", "joint_act"),
                position=attr in ("joint_target_q", "joint_rest_q"),
                **args,
            )
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
                    if attr in ("joint_q", "joint_target_q", "joint_rest_q")
                    else ("m/s" if i < linear else "rad/s")
                )
                if attr in ("joint_f", "joint_act"):
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
