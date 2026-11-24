# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local simulator workspace: bounded inspection work and edge-aligned panels."""

from __future__ import annotations

import math
from time import perf_counter
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp

from ...._version import __version__
from ...geometry.flags import ShapeFlags
from ...geometry.raycast import raycast_kernel
from ...geometry.types import GeoType
from ...sim.articulation import eval_fk, eval_ik
from ...sim.enums import BodyFlags, JointType, ModelFlags
from ...sim.model import Model
from ...sim.state import State
from ...solvers.solver import SolverBase
from ..viewer_attributes import AttributeRows
from ..viewer_bindings import InspectionBindings
from ..viewer_history import FrameHistory, _component_arrays
from ..viewer_model_tree import ModelTree
from . import theme
from .dockspace import ViewerDockspace
from .timeline import TimelineView
from .widgets import PhysicsWidgets

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from numpy.typing import ArrayLike

    from newton._src.viewer.gui.viewer_gui import ViewerGui
    from newton._src.viewer.scene_authoring import SceneEditor
    from newton._src.viewer.viewer_types import TreeKey


@wp.kernel
def _set_shape_opacity(
    opacity: wp.array[float], indices: wp.array[int], values: wp.array[float], previous: wp.array[float]
) -> None:
    i = wp.tid()
    index = indices[i]
    previous[i] = opacity[index]
    opacity[index] = values[i]


@wp.kernel
def _selection_shape_types(
    types: wp.array[int],
    flags: wp.array[int],
    bodies: wp.array[int],
    opacity: wp.array[float],
    show_visual: bool,
    show_collision: bool,
    show_static: bool,
    show_ground: bool,
    output: wp.array[int],
) -> None:
    i = wp.tid()
    visible = (
        (show_visual and (flags[i] & ShapeFlags.VISIBLE) != 0)
        or (show_collision and (flags[i] & ShapeFlags.COLLIDE_SHAPES) != 0)
        or (show_static and bodies[i] < 0)
    )
    if opacity[i] <= 0.0 or (types[i] == GeoType.PLANE and not show_ground):
        visible = False
    output[i] = types[i] if visible else -1


@wp.kernel
def _frame_shape_bounds(
    indices: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    types: wp.array[int],
    parents: wp.array[int],
    local: wp.array[wp.transform],
    bodies: wp.array[wp.transform],
    worlds: wp.array[int],
    offsets: wp.array[wp.vec3],
    layer: wp.transform,
    bounds: wp.array2d[float],
) -> None:
    item = wp.tid()
    shape = indices[item]
    if types[shape] == int(GeoType.PLANE):
        return
    pose = local[shape]
    if parents[shape] >= 0:
        pose = bodies[parents[shape]] * pose
    offset = wp.vec3(0.0)
    world = worlds[shape]
    if world >= 0 and world < offsets.shape[0]:
        offset = offsets[world]
    for corner in range(8):
        point = wp.vec3(
            upper[item][0] if corner & 1 else lower[item][0],
            upper[item][1] if corner & 2 else lower[item][1],
            upper[item][2] if corner & 4 else lower[item][2],
        )
        point = wp.transform_point(layer, wp.transform_point(pose, point) + offset)
        for axis in range(3):
            wp.atomic_min(bounds, 0, axis, point[axis])
            wp.atomic_max(bounds, 1, axis, point[axis])


class ViewerWorkspace:
    """Coordinate ViewerGL panels, selection, and edits for the active layer.

    The application owns the simulation loop and objects. InspectionBindings resolves
    their live bindings, ModelTree indexes topology, and PropertyInspector maps
    attribute metadata to widgets. This class connects those services to ImGui
    and authorizes edits; it does not construct or rebuild Models or solvers.

    Components come only from the viewer's public channels: the setters for
    Control, solver, and collision pipeline, and frame logging for State and
    Contacts. History records those logged frames. Scene-owned caches and history
    are released when the Model changes, while presentation preferences survive.
    """

    def __init__(self, gui: ViewerGui) -> None:
        """Initialize panel presenters, selection and edit state, preferences, and lazy Model caches for one GUI."""
        self.gui = gui
        self.widgets = PhysicsWidgets(gui)
        self.viewer = gui._viewer
        self.history: FrameHistory | None = None
        self._bindings: InspectionBindings | None = None
        self._bindings_components = None
        self.model: Model | None = None
        self.tree: ModelTree | None = None
        self.world = 0
        self.selected: TreeKey | None = None
        self.selection: set[TreeKey] = set()
        self._selection_anchor: TreeKey | None = None
        self._deformable_selection = None
        self._selection_shapes: tuple[int, ...] = ()
        self._contact_filter_shapes: tuple[int, ...] | None = None
        self._contact_filter_masks: dict[Model, wp.array[int]] = {}
        self._frame_bounds_buffers = None
        self._selection_binding_key = None
        self._selection_bindings = ()
        self._joint_selection = None
        self._selection_joints: tuple[int, ...] = ()
        self._joint_selections = None
        self.search = ""
        self.example_search = ""
        self._example_rows = []
        self._example_query = None
        self._example_expanded: set[str] = set()
        self._example_last_current: str | None = None
        self._example_all_folders: tuple[str, ...] = ()
        self._example_stats: tuple[int, int, int] = (0, 0, 0)
        self.left_width = 280.0
        self.right_width = 350.0
        self.timeline_height = TimelineView.default_height
        self.console_height = 250.0
        self._console_pixels = 0.0
        self.show_left = True
        self.show_right = True
        self.rect = (0, 0, 1, 1)
        self.dockspace = ViewerDockspace(self)
        self.authoring = None
        self.example: SceneEditor | None = None
        self._new_scene_engine = 0
        self._cache = {}
        self._live_cache = {}
        self._live_time = 0.0
        self._last_timeline_cursor = None
        self._drafts = {}
        self._undo = None
        self._kinematic_state = None
        self._kinematic_indices = None
        self._joint_values = None
        self.message = ""
        self.draw_ms = 0.0
        self._tree_pending = True
        self._tree_width_key = None
        self._tree_content_width = self._tree_column_width = self._tree_scroll_x = 0.0
        self._tree_scroll_pending = None
        self._hidden_shapes = {}
        self._visibility_cache = {}
        self._drag_undo = None
        self.selection_mode = "body"
        self._selection_ray = None
        self._reveal_selection = False
        self._focus_properties = False
        self._focus_environment = False
        # Custom loops have no example tabs to compete with; show their UI once.
        self._custom_tab_shown = False
        self._focus_scene = False
        self.timeline_options = {
            "recording": False,
            "playback_speed": 1.0,
            "contact_log": True,
            "show_rigid_contacts": True,
            "show_soft_rigid_contacts": True,
            "show_soft_self_contacts": True,
            "show_body_speed": True,
            "show_particle_speed": True,
        }
        self._active_timeline = None
        self._control_owner = None
        self._control_initial: dict[str, wp.array[Any]] = {}
        from .collision import CollisionInspector  # noqa: PLC0415
        from .console import ViewerConsole  # noqa: PLC0415
        from .viewport import ViewportControls  # noqa: PLC0415

        self.console = ViewerConsole(gui.console_capture)
        self.viewport_controls = ViewportControls(self)
        self.collision = CollisionInspector(self)

    @property
    def bindings(self) -> InspectionBindings:
        """Return the components supplied through the active layer's public setters."""
        components = self.viewer._inspection_components
        if self._bindings is None or self._bindings_components is not components:
            self._bindings = InspectionBindings(components=components)
            self._bindings_components = components
        return self._bindings

    def save_preferences(self) -> None:
        """Retain BVH and timeline display options across example or history rebinding in this workspace."""
        if self.tree is not None:
            bvhs = self.tree.resources.bvhs
            self._bvh_options = {name: getattr(bvhs, name) for name in ("show_bounds", "limit")}
        if self._active_timeline is not None:
            for name in self._active_timeline.preference_names:
                if hasattr(self._active_timeline, name):
                    self.timeline_options[name] = getattr(self._active_timeline, name)

    def reset_history(self) -> None:
        """Discard snapshots after an authored edit without disrupting scene interaction."""
        self.save_preferences()
        if authoring := getattr(self, "authoring", None):
            authoring._placement_query = None
        self._set_history(FrameHistory(self.viewer) if self.model is not None else None)
        self._cache.clear()
        self._live_cache.clear()
        self._last_timeline_cursor = None

    def detach(self) -> None:
        """Keep UI choices while releasing scene-owned state and history."""
        self.save_preferences()
        if self.tree is not None:
            self.tree.resources.bvhs.close(self.viewer)
        self.gui.ui.section_errors.clear()
        self._set_history(None)
        self.model = self.tree = None
        self._bindings = self._bindings_components = None
        if inspector := getattr(self, "property_inspector", None):
            inspector.clear()
        self.viewport_controls.clear_scene()
        self._control_owner = None
        self._control_initial.clear()
        self.collision.contacts_tab = False
        self.collision.visible_contacts = ()
        self.viewer.renderer._contact_labels = ()
        self.viewer.renderer._contact_selection = None
        self.collision.tree = None
        self.collision.contacts = self.collision.state = None
        self.collision.pipelines.clear()
        self.collision.cache.clear()
        self._selection_ray = None
        self.select(None)

    def _bind_timeline(self, timeline: TimelineView | None) -> None:
        """Switch timeline presentation, saving old options and restoring them on the newly bound view."""
        if timeline is not self._active_timeline:
            self.save_preferences()
            self._active_timeline = timeline
            for name, value in self.timeline_options.items():
                if hasattr(timeline, name):
                    setattr(timeline, name, value)

    def _set_history(self, history: FrameHistory | None) -> None:
        """Install the history that the viewer records logged frames into, and its timeline view."""
        if history is self.history:
            return
        self.save_preferences()
        self.history = self.viewer._history = history
        self._active_timeline = None
        if history is not None:
            self._bind_timeline(TimelineView(history))

    def bind(self) -> None:
        """Refresh the active scene, preserving preferences across Model changes.

        Called every drawn frame, so unchanged bindings reuse their tree and
        history. A different Model discards topology, selection, history, and
        display buffers before inspection resumes.
        """
        if self.model is not self.viewer.model:
            self._set_history(None)
            if self.tree is not None:
                self.tree.resources.bvhs.close(self.viewer)
            self.model = self.viewer.model
            self._selection_ray = None
            for rotation in getattr(self.gui, "_physics_rotations", {}).values():
                rotation.update(q=None, values=None, error="")
            self._hidden_shapes.clear()
            self._visibility_cache.clear()
            self.tree = None
            self.world = 0
            self.select(None)
            self._tree_pending = True
            if self.model is not None:
                self._set_history(FrameHistory(self.viewer))
        self._capture_control_initial()
        if self.authoring is not None:
            self.authoring.poll()

    def set_scene_editor(self, editor: SceneEditor | None) -> None:
        """Attach an optional authoring presenter without discovering application objects."""
        if editor is self.example:
            return
        self.close_authoring()
        if editor is not None:
            from .authoring import AuthoringPanel  # noqa: PLC0415

            self.authoring = AuthoringPanel(self, editor)
            self.example = editor

    def close_authoring(self) -> None:
        """Release optional authoring services and textures before replacing their owner."""
        if self.authoring is not None:
            self.authoring.close()
            self.authoring = None
        self.example = None

    def notify_model_changed(self, model: Model, flags: ModelFlags | int) -> None:
        """Refresh shape inspection after an explicit Model change notification.

        Geometry variants retain shape indices but can replace their resource
        children. Preserve scene selection and follow mode; if a selected
        resource disappears, select its owning shape instead. Invalidate only
        existing caches, leaving unopened worlds and resource payloads lazy.
        """
        if not flags & (ModelFlags.SHAPE_TYPE_AND_SOURCE | ModelFlags.SHAPE_PROPERTIES):
            return
        # Display modes cover all layers, even when the inspector shows another.
        self.viewport_controls.invalidate_shapes()
        if model is not self.model:
            return
        if authoring := getattr(self, "authoring", None):
            authoring._placement_query = None
        self._cache.clear()
        self._live_cache.clear()
        self._frame_bounds_buffers = None
        self._selection_binding_key = None
        self._navigation = None
        if inspector := getattr(self, "property_inspector", None):
            inspector.clear()
        if self.tree is None:
            return
        removed = self.tree.refresh_shapes()
        if removed.intersection(self.selection):

            def redirect(key: TreeKey) -> TreeKey:
                """Keep the spatial owner when its resource no longer exists."""
                return ("shape", self.tree.resources.shape_owner(key)) if key in removed else key

            selected = redirect(self.selected)
            selection = {redirect(key) for key in self.selection}
            # Reuse normal selection bookkeeping while retaining other selections.
            self.selection = selection - {selected}
            self.select(selected, toggle=True)

    def _ensure_tree(self) -> None:
        """Initialize scene metadata for any consumer, regardless of visible tabs."""
        if self.model is None:
            return
        bindings = self.bindings
        selection = None
        if self.tree is not None and self.tree.bindings is not bindings:
            previous = self.tree.bindings
            if previous.solver is not bindings.solver or previous.collision_pipelines(
                self.model
            ) != bindings.collision_pipelines(self.model):
                selection = sorted(
                    (
                        key
                        for key in self.selection
                        if key[0] in (*self.tree.kinds, "root", "group", "particle", "tri", "tet")
                    ),
                    key=lambda key: (key == self.selected, key[0], repr(key[1])),
                )
                self.tree.resources.bvhs.close(self.viewer)
                self.tree = None
            else:
                self.tree.bindings = self.tree.resources.bvhs.bindings = bindings
        if self.tree is None:
            self.tree = ModelTree(self.model, bindings=bindings)
            for name, value in getattr(self, "_bvh_options", {}).items():
                setattr(self.tree.resources.bvhs, name, value)
        self.collision.attach(self.tree)
        if selection is not None:
            self.select(selection[0] if selection else None)
            for key in selection[1:]:
                self.select(key, toggle=True)

    def select(self, key: TreeKey | None, *, toggle: bool = False, extend: bool = False) -> None:
        """Replace, toggle, extend, or clear selection and invalidate selection-dependent state.

        Range extension follows visible tree rows. Resolve the active object
        and descendants, stop camera following, release old deformable drawing
        resources, and clear property caches and undo state.
        """
        if key is not None:
            self._ensure_tree()
        self.gui.stop_following_selection()
        self._frame_bounds_buffers = None
        if self._deformable_selection is not None:
            self._deformable_selection.close()
            self._deformable_selection = None
        if extend and self.tree is not None and self._selection_anchor is not None:
            keys = [row[0] for row in self.tree.rows]
            if key in keys and self._selection_anchor in keys:
                a, b = sorted((keys.index(key), keys.index(self._selection_anchor)))
                self.selection.update(keys[a : b + 1])
            elif key is not None:
                self.selection.add(key)
        elif toggle and key is not None:
            if key in self.selection:
                self.selection.remove(key)
                # Resource identifiers can mix scalar indices and tuples.
                key = max(self.selection, key=lambda node: (node[0], repr(node[1])), default=None)
            else:
                self.selection.add(key)
            self._selection_anchor = key
        else:
            self.selection = {key} if key is not None else set()
            self._selection_anchor = key
        if key is not None and self.tree is not None:
            resource = self.tree.resources.handles(key)
            if resource:
                self.tree.resources.reveal(self.tree, key)
            if key[0].startswith("attribute:"):
                self.tree.reveal_attribute(key)
            collision = self.collision.handles(key)
            if collision:
                self.tree.meta[key]["world"] = self.collision.world_for(key)
            world = self.tree.world_for(key)
            if not resource and key[0] not in ("actuator_collection", "actuator_group", "battery"):
                self.tree.ensure_world(world)
            if key[0] in ("particle", "tri", "edge", "tet"):
                self.tree.deformables.reveal(self.tree, key)
            elif key[0] in ("tendon_guide", "tendon_segment", "attachment_particle_body", "actuator_group", "battery"):
                self.tree.reveal_aux(key)
            elif self.tree.custom.frequency(key) is not None:
                self.tree.custom.reveal(self.tree, key)
            if (not collision and not resource) or world >= 0:
                self.world = max(0, world)
        self.selected = key
        self._selection_shapes = ()
        self._selection_joints = ()
        shapes, joints = set(), set()
        for item in self.selection if self.model is not None else ():
            kind, index = item
            if kind == "shape":
                shapes.add(index)
            elif kind == "body":
                # Physics ownership is independent of lazy explorer topology.
                shapes.update(self.model.body_shapes.get(index, ()))
            elif kind in ("articulation", "root", "group", "curve") and self.tree is not None:
                shapes.update(self.tree.shape_descendants(item))
                joints.update(self.tree.joint_descendants(item))
            elif kind == "joint":
                joints.add(index)
                child = int(self.model.joint_child[index : index + 1].numpy()[0])
                if child >= 0:
                    shapes.update(self.model.body_shapes.get(child, ()))
            elif kind == "tendon" and self.tree is not None and self.model.tendon_guide_body is not None:
                guides, _ = self.tree.tendon_elements(index)
                if len(guides):
                    bodies = self.model.tendon_guide_body[guides.start : guides.stop].numpy()
                    for body in bodies:
                        if body >= 0:
                            shapes.update(self.model.body_shapes.get(int(body), ()))
            elif kind == "tendon_guide" and self.model.tendon_guide_body is not None:
                if 0 <= index < len(self.model.tendon_guide_body):
                    body = int(self.model.tendon_guide_body[index : index + 1].numpy()[0])
                    if body >= 0:
                        shapes.update(self.model.body_shapes.get(body, ()))
            elif kind == "attachment_particle_body" and self.model.attachment_particle_body_body is not None:
                if 0 <= index < len(self.model.attachment_particle_body_body):
                    body = int(self.model.attachment_particle_body_body[index : index + 1].numpy()[0])
                    if body >= 0:
                        shapes.update(self.model.body_shapes.get(body, ()))
            elif (
                kind == "constraint_mimic"
                and self.model.constraint_mimic_joint0 is not None
                and self.model.constraint_mimic_joint1 is not None
            ):
                if 0 <= index < self.model.constraint_mimic_count:
                    for j_arr in (self.model.constraint_mimic_joint0, self.model.constraint_mimic_joint1):
                        j_idx = int(j_arr[index : index + 1].numpy()[0])
                        if 0 <= j_idx < self.model.joint_count:
                            joints.add(j_idx)
                            child = int(self.model.joint_child[j_idx : j_idx + 1].numpy()[0])
                            if child >= 0:
                                shapes.update(self.model.body_shapes.get(child, ()))
            elif self.tree is not None and (shape := self.tree.resources.shape_owner(item)) is not None:
                shapes.add(shape)
            elif self.collision.handles(item):
                shapes.update(i for i in self.collision.shapes(item) if 0 <= i < self.model.shape_count)
                if kind in ("contact", "soft_contact", "collision_pair"):
                    self.viewer.show_contacts = True
        self._selection_shapes = tuple(sorted(shapes))
        # Contact filtering follows explicit shape/body selections. Other tree
        # objects and direct contact inspection retain the complete contact view.
        owners = [item for item in self.selection if item[0] in ("shape", "body")]
        contact_shapes = set()
        for kind, index in owners:
            contact_shapes.update((index,) if kind == "shape" else self.model.body_shapes.get(index, ()))
        self._contact_filter_shapes = tuple(sorted(contact_shapes)) if owners else None
        self._contact_filter_masks.clear()
        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            camera._navigation_filter = self._camera_motion if shapes else None
        self._navigation = None
        self._selection_joints = tuple(sorted(joints))
        self._joint_selection = None
        self._joint_selections = None
        self._selection_binding_key = None
        self._reveal_selection = False
        self._cache.clear()
        self._live_cache.clear()
        self._drafts.clear()
        self._undo = None
        self._kinematic_state = None
        self._kinematic_indices = None
        self._joint_values = None
        self._drag_undo = None
        self.message = ""
        self._live_time = 0

    def contact_shape_mask(self, model: Model) -> wp.array[int] | None:
        """Cache a device mask per Model until shape/body selection changes.

        A contact matches either witness. Models outside the inspected scene
        receive an empty mask, so coincident shape indices cannot leak contacts.
        No owner selection returns None; a selected empty body matches nothing.
        """
        if self._contact_filter_shapes is None:
            return None
        if model not in self._contact_filter_masks:
            mask = np.zeros(model.shape_count, dtype=np.int32)
            if model is self.model:
                mask[list(self._contact_filter_shapes)] = 1
            self._contact_filter_masks[model] = wp.array(mask, dtype=int, device=model.device)
        return self._contact_filter_masks[model]

    def _camera_motion(self, start: wp.vec3, end: wp.vec3, **kwargs: Any) -> float:
        """Lazily create the selection-proximity filter and return its camera-motion multiplier."""
        if self._navigation is None:
            from ..viewer_navigation import SelectionNavigation  # noqa: PLC0415

            self._navigation = SelectionNavigation(self)
        return self._navigation(start, end, **kwargs)

    def navigate(self, key: TreeKey) -> None:
        """Select a linked object and reveal its hierarchy in the explorer."""
        self.select(key)
        tree = self.tree
        self.search = tree.query = ""
        parent, visited = key, set()
        while parent in tree.parents and parent not in visited:
            visited.add(parent)
            parent = tree.parents[parent]
            tree.expanded.add(parent)
        tree.rebuild()
        self._reveal_selection = self._focus_scene = self._focus_properties = True

    @property
    def can_follow_selection(self) -> bool:
        """Only spatial selections have a persistent identity across simulation steps."""
        return self.selected is not None and self.selected[0] in (
            "body",
            "shape",
            "joint",
            "articulation",
            "root",
            "group",
            "surface",
            "volume",
            "curve",
            "particle_set",
            "elements",
            "particle",
            "tri",
            "edge",
            "tet",
            "tendon",
            "tendon_guide",
        )

    def selection_bounds(self, *, include_contacts: bool = True, sync_selection: bool = True) -> np.ndarray | None:
        """Bound the selected geometry in the displayed state, on demand."""
        state, model, key = self.viewer._last_state, self.model, self.selected
        if key is None or model is None or state is None:
            return None
        points = self.collision.focus_points() if include_contacts else []
        if points:
            return np.array((np.min(points, axis=0) - 0.05, np.max(points, axis=0) + 0.05))
        if self.tree is not None and self.tree.resources.bvhs.handles(key):
            return self.tree.resources.bvhs.bounds(self)
        if sync_selection and self.tree is not None and self.tree.deformables.particle_count:
            self._sync_selection_outline()
        shapes = self._selection_shapes
        offsets = self.viewer.world_offsets
        if shapes:
            batches = getattr(self.viewer, "_shape_to_batch", None)
            frame_key = (id(batches), shapes)
            if self._frame_bounds_buffers is None or getattr(self, "_frame_bounds_key", None) != frame_key:
                lower = np.empty((len(shapes), 3), dtype=np.float32)
                upper = np.empty_like(lower)
                missing = list(range(len(shapes)))
                displayed_bounds = getattr(self.viewer, "_get_shape_framing_bounds", None)
                if displayed_bounds is not None and batches is not None:
                    # Cache the selected display extents once. Following a moving
                    # object only reduces its current poses on the device.
                    local_bounds = {}
                    missing = []
                    slots = self.viewer._shape_to_slot
                    for item, shape in enumerate(shapes):
                        batch = batches[shape]
                        if batch is None:
                            missing.append(item)
                            continue
                        if batch not in local_bounds:
                            local_bounds[batch] = displayed_bounds(batch)
                        bounds_for_batch = local_bounds[batch]
                        if bounds_for_batch is not None:
                            slot = int(slots[shape] - slots[batch.model_shapes[0]])
                            lower[item], upper[item] = (bound[slot] for bound in bounds_for_batch)
                        else:
                            missing.append(item)
                if missing:
                    # Unrendered shapes retain the model-bounds fallback. Restrict
                    # its host readback to the range needed by this selection.
                    indices = np.asarray(shapes)[missing]
                    start, end = int(indices.min()), int(indices.max()) + 1
                    lower[missing] = model.shape_collision_aabb_lower[start:end].numpy()[indices - start]
                    upper[missing] = model.shape_collision_aabb_upper[start:end].numpy()[indices - start]
                self._frame_bounds_buffers = (
                    wp.array(shapes, dtype=int, device=model.device),
                    wp.empty((2, 3), dtype=float, device=model.device),
                    wp.empty(0, dtype=wp.vec3, device=model.device),
                    wp.array(lower, dtype=wp.vec3, device=model.device),
                    wp.array(upper, dtype=wp.vec3, device=model.device),
                )
                self._frame_bounds_key = frame_key
            indices, bounds, empty_offsets, lower, upper = self._frame_bounds_buffers
            bounds[0:1].fill_(1e30)
            bounds[1:2].fill_(-1e30)
            wp.launch(
                _frame_shape_bounds,
                dim=len(shapes),
                inputs=[
                    indices,
                    lower,
                    upper,
                    model.shape_type,
                    model.shape_body,
                    model.shape_transform,
                    state.body_q,
                    model.shape_world,
                    offsets if offsets is not None else empty_offsets,
                    self.viewer.layer.xform,
                ],
                outputs=[bounds],
                device=model.device,
            )
            result = bounds.numpy()
            if np.isfinite(result).all() and (result[1] >= result[0]).all():
                if self._deformable_selection is not None:
                    deformable = self._deformable_selection.bounds()
                    if deformable is not None:
                        result[0] = np.minimum(result[0], deformable[0])
                        result[1] = np.maximum(result[1], deformable[1])
                return result
        if self._deformable_selection is not None:
            result = self._deformable_selection.bounds()
            if result is not None:
                return result
        # Empty bodies and joints still have an inspectable spatial anchor.
        kind, index = key
        pose = None
        if kind == "body":
            pose = wp.transform(*state.body_q[index : index + 1].numpy()[0])
        elif kind in ("shape", "joint"):
            local = model.shape_transform if kind == "shape" else model.joint_X_c
            parent = model.shape_body if kind == "shape" else model.joint_child
            pose = wp.transform(*local[index : index + 1].numpy()[0])
            body = int(parent[index : index + 1].numpy()[0])
            if body >= 0:
                pose = wp.transform(*state.body_q[body : body + 1].numpy()[0]) * pose
        if pose is not None:
            world = int(getattr(model, kind + "_world")[index : index + 1].numpy()[0])
            offset = offsets[world : world + 1].numpy()[0] if offsets is not None and world >= 0 else np.zeros(3)
            center = np.asarray(wp.transform_point(self.viewer.layer.xform, wp.vec3(np.asarray(pose.p) + offset)))
            return np.array((center - 0.05, center + 0.05))
        return None

    def read(self, owner: object, name: str, start: int, count: int = 1, *, live: bool = False) -> np.ndarray:
        """Return a cached host slice, restricting readback to requested storage rows.

        Static metadata is cached until selection or scene invalidation. State
        and Control callers must pass ``live=True``: that cache refreshes on a
        timer, history cursor changes, and edits. Returned arrays may share cache
        storage and must not be used to mutate device data; use ``apply_edit()``.
        """
        array = getattr(owner, name, None)
        if array is None or count <= 0:
            return np.empty(0)
        cache = self._live_cache if live else self._cache
        key = (id(owner), name, start, count)
        if key not in cache:
            # Warp cannot slice empty dimensions, including arrays shaped
            # (world_count, 0). Their NumPy view preserves dtype and dimensions
            # without transferring any device data.
            cache[key] = (
                array.numpy()[start : start + count] if not array.size else array[start : start + count].numpy()
            )
        return cache[key]

    def _sync_selection_outline(self) -> None:
        """Resolve selected shapes and deformables to current renderer instances.

        Cache shape-to-batch mappings until selection or batch identity changes;
        resolve object handles each frame because display changes can replace
        buffers. Also install or remove the selection-based navigation filter.
        """
        viewer = self.viewer
        batches = viewer._shape_to_batch
        sdf_batches = viewer._sdf_isomesh_instances
        binding_key = (id(batches), id(sdf_batches), len(sdf_batches), self._selection_shapes)
        if binding_key != self._selection_binding_key:
            bindings = []
            if batches is not None:
                slots = viewer._shape_to_slot
                for shape in self._selection_shapes:
                    batch = batches[shape]
                    if batch is None:
                        continue
                    index = int(slots[shape] - slots[batch.model_shapes[0]])
                    bindings.append((batch.name, index))
            # Collision display can replace the visual mesh with its SDF
            # isosurface. Cache its instance mapping only when selection or
            # batches change, without reading GPU data during rendering.
            if self._selection_shapes:
                for batch in sdf_batches.values():
                    for shape in self._selection_shapes:
                        bindings.extend(
                            (batch.name, int(index)) for index in np.flatnonzero(batch.model_shapes == shape)
                        )
            self._selection_bindings = bindings
            self._selection_binding_key = binding_key
        # Resolve names each frame because a batch's buffers can be replaced
        # when opacity, world visibility, or display options change.
        viewer.renderer._selection_instances = tuple(
            (obj, index)
            for name, index in self._selection_bindings
            if (obj := viewer.objects.get(name)) is not None and not obj.hidden
        )
        viewer.renderer._selection_scale = self.widgets.scale
        viewer.log_points("/workspace/selected_particles", points=None, hidden=True)
        if self.tree is not None:
            candidates = set(self.selection)
            for key in self.selection:
                if key[0] in ("root", "group"):
                    candidates.update(self.tree.deformable_descendants(key))
            keys = [
                key
                for key in candidates
                if key[0] in ("surface", "volume", "particle_set", "elements", "particle", "tri", "tet")
                and viewer._should_render_world(self.tree.world_for(key))
                and self.tree.deformables.visible(self.tree, key)
            ]
            signature = tuple(sorted(keys))
            if self._deformable_selection is not None and signature != self._deformable_selection.keys:
                self._deformable_selection.close()
                self._deformable_selection = None
            if keys:
                if self._deformable_selection is None:
                    from ..viewer_deformables import DeformableSelection  # noqa: PLC0415

                    self._deformable_selection = DeformableSelection(self, keys)
                    self._deformable_selection.keys = signature
                self._deformable_selection.update(self)
        camera = getattr(viewer, "camera", None)
        if camera is not None:
            camera._navigation_filter = (
                self._camera_motion if self._selection_shapes or self._deformable_selection else None
            )

    def select_from_screen(
        self,
        x: float,
        y: float,
        to_framebuffer_coords: Callable[[float, float], tuple[float, float]],
        *,
        shape: bool = False,
        toggle: bool = False,
    ) -> None:
        """Select visible geometry without activating force picking or changing physics."""
        viewer, model = self.viewer, self.viewer.model
        state = viewer._last_state
        if model is None or state is None:
            return
        self.bind()
        px, py = to_framebuffer_coords(x, y)
        rect = viewer.renderer._workspace_rect
        if rect is not None:
            if not (rect[0] <= px < rect[0] + rect[2] and rect[1] <= py < rect[1] + rect[3]):
                return
            px, py = px - rect[0], py - rect[1]
        origin, direction = viewer.camera.get_world_ray(float(px), float(py))
        inverse = wp.transform_inverse(viewer.layer.xform)
        origin = wp.transform_point(inverse, wp.vec3(*origin))
        direction = wp.transform_vector(inverse, wp.vec3(*direction))
        self._ensure_tree()
        if self._selection_ray is None:
            self._selection_ray = SimpleNamespace(
                types=wp.empty(model.shape_count, dtype=int, device=model.device),
                distance=wp.empty(1, dtype=float, device=model.device),
                shape=wp.empty(1, dtype=int, device=model.device),
                body=wp.empty(1, dtype=int, device=model.device),
                lock=wp.zeros(1, dtype=int, device=model.device),
                offsets=wp.empty(0, dtype=wp.vec3, device=model.device),
            )
        ray = self._selection_ray
        ray.distance.fill_(1e10)
        ray.shape.fill_(-1)
        ray.body.fill_(-1)
        ray.lock.zero_()
        wp.launch(
            _selection_shape_types,
            dim=model.shape_count,
            inputs=[
                model.shape_type,
                model.shape_flags,
                model.shape_body,
                model.shape_opacity,
                viewer.show_visual,
                viewer.show_collision,
                viewer.show_static,
                viewer.show_ground,
            ],
            outputs=[ray.types],
            device=model.device,
        )
        wp.launch(
            raycast_kernel,
            dim=model.shape_count,
            inputs=[
                state.body_q,
                model.shape_body,
                model.shape_transform,
                ray.types,
                model.shape_scale,
                model.shape_source_ptr,
                origin,
                direction,
                ray.lock,
                ray.distance,
                ray.shape,
                ray.body,
                model.shape_world,
                viewer.world_offsets if viewer.world_offsets is not None else ray.offsets,
                viewer._visible_worlds_mask,
                # Null arrays disable variant filtering for ordinary models.
                model._shape_variant_row,
                model._shape_variant_shape,
            ],
            device=model.device,
        )
        index = int(ray.shape.numpy()[0])
        deformable = (
            self.tree.deformables.pick(
                self,
                origin,
                direction,
                particles=self.selection_mode == "particle" or not model.tri_count,
                limit=float(ray.distance.numpy()[0]),
                triangle=shape or self.selection_mode == "triangle",
            )
            if self.tree.deformables.particle_count
            else None
        )
        if index < 0 and deformable is None:
            if not toggle:
                self.select(None)
            return
        body = int(ray.body.numpy()[0])
        world = (
            self.tree.world_for(deformable)
            if deformable is not None
            else int(model.shape_world[index : index + 1].numpy()[0])
        )
        self.tree.ensure_world(world)
        key = (
            deformable
            if deformable is not None
            else ("shape", index)
            if shape or self.selection_mode == "shape" or body < 0
            else ("body", body)
        )
        self.select(key, toggle=toggle)
        self.search = self.tree.query = ""
        parent, seen = key, set()
        while parent in self.tree.parents and parent not in seen:
            seen.add(parent)
            parent = self.tree.parents[parent]
            self.tree.expanded.add(parent)
        self.tree.rebuild()
        self._reveal_selection = self._focus_scene = self._focus_properties = True
        self.show_right = True

    def _window(self, name: str, x: float, y: float, w: float, h: float, *, flags: int = 0) -> bool:
        """Begin a fixed-position workspace window; the caller must call ImGui end() even when it returns False."""
        ui = self.gui.ui.imgui
        ui.set_next_window_pos(ui.ImVec2(x, y))
        ui.set_next_window_size(ui.ImVec2(max(w, 1), max(h, 1)))
        flags |= (
            ui.WindowFlags_.no_title_bar
            | ui.WindowFlags_.no_move
            | ui.WindowFlags_.no_resize
            | ui.WindowFlags_.no_saved_settings
            | ui.WindowFlags_.no_docking
        )
        return ui.begin(name, flags=flags)[0]

    def draw(self) -> None:
        """Coordinate workspace layout, inspector refreshes, and viewport overlays for one UI frame.

        Throttle live property readbacks, but invalidate immediately on paused
        history scrubbing. Convert the docked viewport rectangle to framebuffer
        coordinates before synchronizing renderer overlays.
        """
        started = perf_counter()
        ui = self.gui.ui.imgui
        self.console.poll_warnings()
        self.bind()
        timeline = self.history
        cursor = (id(timeline), timeline.index) if timeline is not None else None
        # Property readouts need not synchronize the device at render frequency.
        # Scrubbing while paused must bypass that throttle to show the chosen frame.
        if started - self._live_time > 0.15 or (self.viewer.is_paused() and cursor != self._last_timeline_cursor):
            self._live_cache.clear()
            self._live_time = started
        self._last_timeline_cursor = cursor
        s = self.gui.ui.dpi_scale
        w, h = ui.get_io().display_size
        top, status = 34 * s, 28 * s
        active_timeline = self._active_timeline
        timeline_expanded = active_timeline is not None and active_timeline.has_track
        bottom = min(self.timeline_height * s, h * 0.36) if timeline_expanded else TimelineView.compact_height * s
        left = min(self.left_width * s, w * 0.28) if self.show_left else 0
        right = min(self.right_width * s, w * 0.31) if self.show_right and w > 740 * s else 0
        panel_h = max(40, h - top - bottom - status)
        target = min(self.console_height * s, panel_h * 0.65) if self.console.open else 0.0
        delta = min(ui.get_io().delta_time, 0.05)
        self._console_pixels += (target - self._console_pixels) * (1.0 - math.exp(-22.0 * delta))
        if abs(target - self._console_pixels) < 0.5:
            self._console_pixels = target
        console_h = round(min(self._console_pixels, panel_h * 0.65))
        view_h = max(40, panel_h - console_h)
        dockspace_rect = (int(left), int(top), int(max(1, w - left - right)), int(view_h))
        self.rect = self.dockspace.draw(dockspace_rect)
        renderer = self.viewer.renderer
        scale_x, scale_y = renderer._screen_width / max(1, w), renderer._screen_height / max(1, h)
        x, y, width, height = self.rect
        framebuffer_rect = (
            round(x * scale_x),
            round((h - y - height) * scale_y),
            max(1, round(width * scale_x)),
            max(1, round(height * scale_y)),
        )
        self.viewer.renderer._workspace_rect = framebuffer_rect
        self.viewer.camera.width, self.viewer.camera.height = framebuffer_rect[2:]
        if self._window("##workspace_header", 0, 0, w, top):
            self.gui.ui.draw_section("Toolbar", self._header)
        ui.end()
        if left:
            if self._window("##workspace_explorer", 0, top, left, view_h):
                self.gui.ui.draw_section(
                    "Explorer", self.authoring.draw if self.authoring is not None else self._explorer
                )
            ui.end()
        # Visibility is established by the inspector drawn this frame, not by
        # the last selected tab in a hidden panel or the Environment/Solver tab.
        self.collision.contacts_tab = False
        if right:
            if self._window("##workspace_inspector", w - right, top, right, view_h):
                self.gui.ui.draw_section(
                    "Properties", self.authoring.draw_inspector if self.authoring is not None else self._inspector
                )
            ui.end()
        if console_h > 4 * s:
            if self._window("##workspace_console", 0, top + view_h, w, console_h):
                if console_h > 65 * s:
                    self.gui.ui.draw_section("Console", lambda: self.console.draw(self))
            ui.end()
            self._splitter(
                "console",
                0,
                top + view_h - 3 * s,
                w,
                6 * s,
                "console_height",
                -1,
                120,
                700,
            )
        timeline = self._active_timeline
        timeline_flags = (
            0 if timeline_expanded else (ui.WindowFlags_.no_scrollbar | ui.WindowFlags_.no_scroll_with_mouse)
        )
        if self._window("##workspace_timeline", 0, h - bottom - status, w, bottom, flags=timeline_flags):
            if timeline is not None:
                self._bind_timeline(timeline)
                self.gui.ui.draw_section("Timeline", lambda: timeline.draw_gui(ui, self.viewer, embedded=True))
            else:
                self._transport()
                ui.same_line(spacing=10 * s)
                self.widgets.label("Select an example to begin.", height=22 * s, muted=True)
        ui.end()
        if self._window(
            "##workspace_status",
            0,
            h - status,
            w,
            status,
            flags=ui.WindowFlags_.no_scrollbar | ui.WindowFlags_.no_scroll_with_mouse,
        ):
            self.gui.ui.draw_section("Status", lambda: self._status(timeline))
        ui.end()
        self._splitter("left", left - 3 * s, top, 6 * s, view_h, "left_width", 1, 190, 440) if left else None
        self._splitter("right", w - right - 3 * s, top, 6 * s, view_h, "right_width", -1, 250, 500) if right else None
        if timeline_expanded:
            self._splitter(
                "bottom",
                0,
                h - bottom - status - 3 * s,
                w,
                6 * s,
                "timeline_height",
                -1,
                TimelineView.default_height,
                380,
            )
        self._sync_selection_outline()
        if self.dockspace.viewport_visible:
            self.viewport_controls.draw()
            self.collision.overlays()
        if self.tree is not None and self.dockspace.viewport_visible:
            self.tree.resources.bvhs.overlay(self)
        self.viewer.renderer._joint_triangles = None
        if self._selection_joints and self.dockspace.viewport_visible:
            if self._joint_selections is None:
                from ..viewer_joint_selection import JointSelectionGroup  # noqa: PLC0415

                self._joint_selections = JointSelectionGroup(self, self._selection_joints)
                if len(self.selection) == 1 and self.selected[0] == "joint":
                    self._joint_selection = self._joint_selections.joints[0]
            self._joint_selections.draw(self, compact=len(self.selection) > 1 or self.selected[0] != "joint")
        if self.authoring is not None and self.dockspace.viewport_visible:
            self.authoring.prepare_gizmo()
        self.draw_ms = 0.9 * self.draw_ms + 0.1 * ((perf_counter() - started) * 1000)

    def _status(self, timeline: TimelineView | None) -> None:
        """Sample live metrics at 5 Hz and reserve stable slots for their values."""
        ui, scale, model = self.gui.ui.imgui, self.widgets.scale, self.model
        self.console.status_button(self)
        paused = self.viewer.is_paused()
        history = timeline.history if isinstance(timeline, TimelineView) else timeline
        key = (id(model), id(timeline), id(getattr(history, "frames", None)), paused)
        if paused and history is not None:
            key += (history.first_frame, history.index, len(history.frames))
        now = perf_counter()
        if key != getattr(self, "_status_key", None) or now - self._status_time >= 0.2:
            items = timeline.status_items() if timeline is not None else []
            if model is not None:
                counts = " · ".join(
                    f"{count:,} {singular if count == 1 else plural}"
                    for singular, plural, count in (
                        ("world", "worlds", model.world_count),
                        ("body", "bodies", model.body_count),
                        ("shape", "shapes", model.shape_count),
                        ("joint", "joints", model.joint_count),
                    )
                )
                items.append(("Scene", counts, theme.STATUS_DETAIL[:3]))
            self._status_items = items
            self._status_perf = f"{self.gui._current_fps:.0f}", f"{self.draw_ms:.1f} ms"
            self._status_key, self._status_time = key, now
        items = self._status_items
        origin = ui.get_cursor_screen_pos()
        available = ui.get_content_region_avail().x
        font = getattr(self.gui.ui, "number_font", None)
        draw = ui.get_window_draw_list()
        center_y = ui.get_window_pos().y + ui.get_window_size().y * 0.5
        font_key = (ui.get_font().font_id, font.font_id if font else None, scale)
        if font_key != getattr(self, "_status_font_key", None):
            self._status_text_centers = {}
            self._status_font_key = font_key
        text_centers = self._status_text_centers
        templates = {
            "Step": "999.99 ms",
            "Depth": "-999.99 mm",
            "History": "9999.9 MiB",
            "Body speed": "9.99e+99 m/s",
            "Particle speed": "9.99e+99 m/s",
            "FPS": "99999",
            "UI": "999.9 ms",
        }

        def centered_text(x: float, text: str, color: int, *, numeric: bool = False) -> None:
            # Different font families have different internal top padding.
            # Align visible glyphs, bypassing ImGui's previous-item baseline.
            # Fixed digit metrics keep changing values at a stable height.
            sample = "0123456789" if numeric else text
            key = (ui.get_font().font_id, ui.get_font_size(), sample)
            if key not in text_centers:
                baked = ui.get_font_baked()
                glyphs = [baked.find_glyph(ord(char)) for char in sample if not char.isspace()]
                text_centers[key] = (min(g.y0 for g in glyphs) + max(g.y1 for g in glyphs)) * 0.5 if glyphs else 0
            draw.add_text(ui.ImVec2(x, round(center_y - text_centers[key])), color, text)

        def value_width(label: str, value: ArrayLike) -> float:
            if label == "Scene":
                return ui.calc_text_size(value).x
            if font:
                ui.push_font(font, 13.0)
            width = ui.calc_text_size(templates.get(label, "9999.9M")).x + 2 * scale
            if font:
                ui.pop_font()
            return width

        def metric(x: float, label: str, value: ArrayLike, rgb: ArrayLike) -> float:
            centered_text(x, label, ui.get_color_u32(ui.Col_.text_disabled))
            value_x = x + ui.calc_text_size(label).x + 6 * scale
            width = value_width(label, value)
            if font and label != "Scene":
                ui.push_font(font, 13.0)
            # Unusual magnitudes must never overlap the next reserved slot.
            draw.push_clip_rect(
                ui.ImVec2(value_x, ui.get_window_pos().y),
                ui.ImVec2(value_x + width, ui.get_window_pos().y + ui.get_window_size().y),
                True,
            )
            centered_text(value_x, value, ui.get_color_u32(ui.ImVec4(*rgb, 1)), numeric=label != "Scene")
            draw.pop_clip_rect()
            if font and label != "Scene":
                ui.pop_font()
            ui.set_cursor_screen_pos(ui.ImVec2(x, ui.get_window_pos().y + 2 * scale))
            ui.invisible_button("##status_" + label, ui.ImVec2(value_x + width - x, ui.get_window_size().y - 4 * scale))
            if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
                detail = {
                    "Step": "Simulation time advanced per displayed frame; includes all solver substeps.",
                    "Rigid contacts": "Rigid contact candidates in this frame, including separated or unloaded contacts. Filled mode disks indicate measured contact force.",
                    "Soft contacts": "Deformable self-contact candidates in this frame.",
                    "Soft-rigid contacts": "Contact candidates between deformable features and rigid shapes in this frame.",
                    "Depth": "Contact penetration depth in the selected frame.",
                    "States": "Recorded simulation states available in the timeline.",
                    "History": "Memory used by recorded states, contacts and solver observables.",
                    "Body speed": "Mean linear body speed in the selected frame, in metres per second.",
                    "Particle speed": "Mean particle speed in the selected frame, in metres per second.",
                    "Scene": "Total model counts across all worlds, including hidden objects.",
                    "FPS": "Viewer frames rendered per second.",
                    "UI": "Smoothed CPU time spent drawing the workspace interface per frame.",
                }.get(label, "")
                ui.set_tooltip(f"{label}: {value}\n{detail}")
            return value_x + width + 18 * scale

        perf = [("FPS", self._status_perf[0], theme.STATUS_DETAIL[:3])]
        if available > 1000 * scale:
            perf.append(("UI", self._status_perf[1], theme.STATUS_DETAIL[:3]))
        perf_width = (
            sum(
                ui.calc_text_size(label).x + 6 * scale + value_width(label, value) + 18 * scale
                for label, value, _ in perf
            )
            - 18 * scale
        )
        perf_x = origin.x + available - perf_width
        x = origin.x
        ui.dummy(ui.ImVec2(available, ui.get_font_size()))
        for i, (label, value, rgb) in enumerate(items):
            width = ui.calc_text_size(label).x + 6 * scale + value_width(label, value)
            if x + width > perf_x - 28 * scale:
                ui.set_cursor_screen_pos(ui.ImVec2(x, origin.y))
                ui.invisible_button("##status_overflow", ui.ImVec2(ui.calc_text_size("…").x, ui.get_font_size()))
                centered_text(x, "…", ui.get_color_u32(ui.Col_.text_disabled))
                if ui.is_item_hovered():
                    ui.set_tooltip("\n".join(f"{label}: {value}" for label, value, _ in items[i:]))
                break
            x = metric(x, label, value, rgb)
        for label, value, rgb in perf:
            perf_x = metric(perf_x, label, value, rgb)

    def _splitter(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        attribute: str,
        direction: float,
        minimum: float,
        maximum: float,
    ) -> None:
        """Draw a panel resize handle and update its logical dimension within the supplied bounds."""
        ui = self.gui.ui.imgui
        ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(0, 0))
        ui.push_style_var(ui.StyleVar_.window_min_size, ui.ImVec2(1, 1))
        ui.push_style_var(ui.StyleVar_.window_border_size, 0.0)
        ui.push_style_color(ui.Col_.window_bg, ui.ImVec4(*theme.TRANSPARENT))
        if self._window("##split_" + name, x, y, w, h):
            ui.invisible_button("##handle", ui.ImVec2(w, h))
            if ui.is_item_hovered() or ui.is_item_active():
                ui.set_mouse_cursor(
                    ui.MouseCursor_.resize_ns if name in ("bottom", "console") else ui.MouseCursor_.resize_ew
                )
            if ui.is_item_active():
                delta = ui.get_io().mouse_delta.y if name in ("bottom", "console") else ui.get_io().mouse_delta.x
                value = getattr(self, attribute) + direction * delta / self.gui.ui.dpi_scale
                setattr(self, attribute, max(minimum, min(maximum, value)))
        ui.end()
        ui.pop_style_color()
        ui.pop_style_var(3)

    def _header(self) -> None:
        """Draw panel, framing, selection, reset, and selection-mode actions plus the version label."""
        ui = self.gui.ui.imgui
        if self.widgets.icon_button("##explorer", "tree", "Explorer", active=self.show_left, size=22):
            self.show_left = not self.show_left
        ui.same_line()
        if self.widgets.icon_button("##inspector", "inspector", "Properties", active=self.show_right, size=22):
            self.show_right = not self.show_right
        ui.same_line()
        if self.widgets.icon_button(
            "##frame_scene", "focus", "Frame selection (F)" if self.selected else "Frame scene (F)", size=22
        ):
            self.gui.frame_camera_on_model()
        ui.same_line()
        if self.widgets.icon_button(
            "##deselect", "deselect", "Deselect all", enabled=bool(self.selection), show_lock=False, size=22
        ):
            self.select(None)
        ui.same_line()
        if self.widgets.icon_button("##reset_layout", "grid", "Reset panel layout", size=22):
            self.left_width, self.right_width, self.timeline_height = 280, 350, TimelineView.default_height
            self.show_left = self.show_right = True
            self.dockspace.reset()
        ui.same_line()
        self.dockspace.menu()
        reset = getattr(self.viewer, "_reset_callback", None)
        if reset:
            ui.same_line()
            if self.widgets.icon_button("##restart", "reset", "Restart simulation", size=22):
                reset()
        for mode, icon in (("body", "body"), ("shape", "mesh"), ("triangle", "tri"), ("particle", "particles")):
            ui.same_line()
            if self.widgets.icon_button(
                "##select_" + mode,
                icon,
                f"Select {'objects' if mode == 'body' else mode + 's'}. Ctrl-click toggles selection; Alt-click selects a shape or triangle.",
                active=self.selection_mode == mode,
                size=22,
            ):
                self.selection_mode = mode
        browser = getattr(self.gui, "_workspace_browser", None)
        if browser is not None:
            ui.same_line()
            if self.widgets.button("New scene"):
                self._new_scene_engine = 0
                ui.open_popup("Create new scene##new_scene")
        version = f"Newton {__version__}"
        left = ui.get_item_rect_max().x + 16 * self.widgets.scale
        pos, size = ui.get_window_pos(), ui.get_window_size()
        right = pos.x + size.x - ui.get_style().window_padding.x
        ui.push_font(self.gui.ui.io.font_default, 14.0)
        draw = ui.get_window_draw_list()
        draw.push_clip_rect(ui.ImVec2(left, pos.y), ui.ImVec2(max(left, right), pos.y + size.y), True)
        draw.add_text(
            ui.ImVec2(max(left, right - ui.calc_text_size(version).x), self.widgets.text_top(pos.y, size.y)),
            self.widgets.color(theme.VERSION_TEXT),
            version,
        )
        draw.pop_clip_rect()
        ui.pop_font()
        if (
            browser is not None
            and ui.begin_popup_modal("Create new scene##new_scene", None, ui.WindowFlags_.always_auto_resize)[0]
        ):
            engines = [("MuJoCo (CPU)", "mujoco", "cpu")]
            if wp.is_cuda_available():
                device = self.viewer.device if self.viewer.device.is_cuda else wp.get_cuda_device()
                engines.append(("MuJoCo Warp (GPU)", "mujoco", device.alias))
            engines.append(("XPBD (CPU)", "xpbd", "cpu"))
            engines.append(("VBD deformables (CPU)", "vbd", "cpu"))
            if wp.is_cuda_available():
                engines.append(("VBD deformables (GPU)", "vbd", device.alias))
                engines.append(("MPM particles (GPU)", "mpm", device.alias))
            _, self._new_scene_engine = ui.combo("Physics", self._new_scene_engine, [item[0] for item in engines])
            _, solver, device = engines[self._new_scene_engine]
            if solver == "mujoco" and device == "cpu":
                ui.text_unformatted("Fast for small rigid scenes; mesh collisions use convex hulls.")
            elif solver == "mujoco":
                ui.text_unformatted("Preserves Newton mesh contacts; requires CUDA.")
            elif solver == "vbd":
                ui.text_unformatted("Cloth, cables, soft bodies, and rigid props with Newton VBD.")
            elif solver == "mpm":
                ui.text_unformatted("Paint particles while paused; rigid props remain stationary. Requires CUDA.")
            else:
                ui.text_unformatted("Newton XPBD solver on CPU.")
            if self.authoring is not None and self.authoring.editor.objects:
                ui.separator()
                ui.text_unformatted("The current scene will close. Save it as USD first to keep it.")
            if self.widgets.button("Cancel"):
                ui.close_current_popup()
            ui.same_line()
            if self.widgets.button("Create new scene"):
                browser.new_scene(solver=solver, device=device)
                ui.close_current_popup()
            ui.end_popup()

    def _transport(self) -> None:
        """Draw basic play, pause, single-step, and restart controls when no history timeline is used."""
        ui = self.gui.ui.imgui
        if self.widgets.icon_button(
            "##workspace_play",
            "play" if self.viewer.is_paused() else "pause",
            "Play" if self.viewer.is_paused() else "Pause",
            size=22,
            prominent=True,
        ):
            self.viewer._paused = not self.viewer._paused
            self.viewer._step_requested = False
        ui.same_line()
        if self.widgets.icon_button("##workspace_step", "play", "Next physics frame", size=22):
            self.viewer._paused = True
            self.viewer._step_requested = True
        reset = getattr(self.viewer, "_reset_callback", None)
        if reset:
            ui.same_line()
            if self.widgets.button("Restart"):
                reset()

    def _explorer(self) -> None:
        """Draw Scene and Examples tabs, filtering and clipping browser rows before dispatching example selection."""
        ui = self.gui.ui.imgui
        if not ui.begin_tab_bar("##explorer_tabs"):
            return
        if self.widgets.tab_item("Scene", flags=ui.TabItemFlags_.set_selected if self._focus_scene else 0)[0]:
            self._focus_scene = False
            self._scene_tree()
            ui.end_tab_item()
        tree = self.gui._example_tree
        if tree and self.widgets.tab_item("Examples")[0]:
            self._example_browser(tree)
            ui.end_tab_item()
        ui.end_tab_bar()

    @staticmethod
    def _example_folder_parts(category: str, module_path: str) -> tuple[str, ...]:
        """Return folder segments for an example from its module path or category."""
        if module_path.startswith("newton.examples."):
            parts = tuple(part for part in module_path.split(".")[2:-1] if part)
            if parts:
                return parts
        parts = tuple(part for part in category.split("/") if part)
        return parts or (category,)

    @staticmethod
    def _example_display_name(folder: str, name: str) -> str:
        """Trim a redundant leading ``<folder>_`` prefix for display inside that folder."""
        prefix = f"{folder}_"
        return name[len(prefix) :] if name.startswith(prefix) and len(name) > len(prefix) else name

    def _rebuild_example_rows(self, tree: dict[str, list[tuple[str, str]]]) -> None:
        """Build clipped folder/example tree rows reflecting the directory hierarchy and search filter."""
        query = self.example_search.strip().casefold()
        current = self.gui._example_current
        root: dict[str, Any] = {"folders": {}, "examples": []}
        total_examples = 0
        current_ancestors: list[str] = []

        for category, entries in sorted(tree.items()):
            for name, path in entries:
                total_examples += 1
                parts = self._example_folder_parts(category, path)
                node = root
                prefix_parts: list[str] = []
                for part in parts:
                    prefix_parts.append(part)
                    folder_key = "/".join(prefix_parts)
                    child = node["folders"].get(part)
                    if child is None:
                        child = {"key": folder_key, "name": part, "folders": {}, "examples": []}
                        node["folders"][part] = child
                    node = child
                display = self._example_display_name(parts[-1], name)
                node["examples"].append((name, display, path, "/".join(parts)))
                if path == current:
                    current_ancestors = ["/".join(parts[: i + 1]) for i in range(len(parts))]

        if current != self._example_last_current:
            self._example_last_current = current
            if current_ancestors:
                self._example_expanded.update(current_ancestors)

        all_folders: list[str] = []

        def collect_rows(node: dict[str, Any], depth: int) -> tuple[list[tuple[str, int, str, str, Any]], int]:
            rows: list[tuple[str, int, str, str, Any]] = []
            matched = 0
            for folder_name in sorted(node["folders"]):
                child = node["folders"][folder_name]
                all_folders.append(child["key"])
                child_rows, child_count = collect_rows(child, depth + 1)
                if query and child_count == 0:
                    continue
                matched += child_count
                rows.append(("folder", depth, folder_name, child["key"], child_count))
                if query or child["key"] in self._example_expanded:
                    rows.extend(child_rows)
            for name, display, path, folder_path in sorted(node["examples"], key=lambda item: item[1]):
                if query and query not in f"{folder_path} {name} {display}".casefold():
                    continue
                matched += 1
                rows.append(("example", depth, display, path, name))
            return rows, matched

        rows, matched_examples = collect_rows(root, 0)
        self._example_rows = rows
        self._example_all_folders = tuple(all_folders)
        self._example_stats = (len(all_folders), matched_examples, total_examples)
        self._example_query = (id(tree), self.example_search, frozenset(self._example_expanded), current)

    def _example_browser(self, tree: dict[str, list[tuple[str, str]]]) -> None:
        """Draw the Examples tab as a searchable, clipped folder tree matching the Scene explorer."""
        ui, widgets = self.gui.ui.imgui, self.widgets
        ui.set_next_item_width(-1)
        _, self.example_search = ui.input_text_with_hint(
            "##example_search", "Search examples or folders...", self.example_search
        )
        current = self.gui._example_current
        cache_key = (id(tree), self.example_search, frozenset(self._example_expanded), current)
        if cache_key != self._example_query:
            self._rebuild_example_rows(tree)

        query = self.example_search.strip()
        if not query:
            if self._example_expanded:
                if widgets.button("Collapse", small=True):
                    self._example_expanded.clear()
                    self._rebuild_example_rows(tree)
            elif widgets.button("Expand", small=True):
                self._example_expanded.update(self._example_all_folders)
                self._rebuild_example_rows(tree)
            ui.same_line()

        folder_count, matched_count, total_count = self._example_stats
        if query:
            noun = "example" if total_count == 1 else "examples"
            ui.text_disabled(f"{matched_count:,} / {total_count:,} {noun}")
        else:
            folders = "folder" if folder_count == 1 else "folders"
            examples = "example" if total_count == 1 else "examples"
            ui.text_disabled(f"{folder_count:,} {folders} · {total_count:,} {examples}")
        ui.separator()

        if ui.begin_child("##example_rows"):
            table_flags = ui.TableFlags_.row_bg | ui.TableFlags_.sizing_stretch_prop
            if ui.begin_table("##example_columns", 2, table_flags):
                ui.table_setup_column("Example", ui.TableColumnFlags_.width_stretch)
                ui.table_setup_column("Count", ui.TableColumnFlags_.width_fixed, 34 * widgets.scale)
                ui.table_headers_row()
                clipper = ui.ListClipper()
                rows = self._example_rows
                clipper.begin(len(rows))
                while clipper.step():
                    for row in range(clipper.display_start, clipper.display_end):
                        self._example_tree_row(*rows[row], filtering=bool(query))
                clipper.end()
                ui.end_table()
        ui.end_child()

    def _example_tree_row(self, kind: str, depth: int, label: str, value: str, meta: Any, *, filtering: bool) -> None:
        """Draw one folder or example row in the Examples tree table."""
        ui, widgets = self.gui.ui.imgui, self.widgets
        ui.push_id(f"{kind}:{value}")
        ui.table_next_row()
        ui.table_next_column()
        indent = depth * 12 * widgets.scale
        ui.set_cursor_pos_x(ui.get_cursor_pos_x() + indent)
        pos = ui.get_cursor_screen_pos()

        is_folder = kind == "folder"
        active = not is_folder and value == self.gui._example_current
        flags = (
            ui.TreeNodeFlags_.no_tree_push_on_open
            | ui.TreeNodeFlags_.span_avail_width
            | ui.TreeNodeFlags_.frame_padding
        )
        if not is_folder:
            flags |= ui.TreeNodeFlags_.leaf
        if active:
            flags |= ui.TreeNodeFlags_.selected
            ui.table_set_bg_color(ui.TableBgTarget_.row_bg0, widgets.color(theme.TREE_SELECTED))
            ui.push_style_color(ui.Col_.header, ui.ImVec4(*theme.TRANSPARENT))
            ui.push_style_color(ui.Col_.text, ui.ImVec4(*widgets.accent))
        if is_folder:
            ui.set_next_item_open(filtering or value in self._example_expanded, ui.Cond_.always)
        ui.tree_node_ex("        " + label + "##node", flags)
        if active:
            ui.pop_style_color(2)

        row_height = ui.get_item_rect_max().y - ui.get_item_rect_min().y
        if is_folder:
            if not filtering and ui.is_item_toggled_open():
                if value in self._example_expanded:
                    self._example_expanded.discard(value)
                else:
                    self._example_expanded.add(value)
        elif ui.is_item_clicked():
            self.gui.select_example(value)

        icon_pos = (pos.x + ui.get_tree_node_to_label_spacing(), pos.y + (row_height - 16 * widgets.scale) / 2)
        widgets.icon(
            "folder" if is_folder else "play",
            icon_pos,
            16 * widgets.scale,
            active=active,
            tint=widgets.explorer_colors.get("group") if is_folder else widgets.explorer_colors.get("root"),
            solid=is_folder or active,
        )
        if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            if is_folder:
                noun = "example" if meta == 1 else "examples"
                ui.set_tooltip(f"newton/examples/{value} ({meta} {noun})")
            else:
                ui.set_tooltip(f"{meta}\n{value}")

        ui.table_next_column()
        if is_folder:
            text = str(meta)
            avail = ui.get_content_region_avail().x
            width = ui.calc_text_size(text).x
            if avail > width:
                ui.set_cursor_pos_x(ui.get_cursor_pos_x() + avail - width)
            ui.text_disabled(text)
        ui.pop_id()

    def _scene_tree(self) -> None:
        """Draw searchable, clipped tree rows with independent name scrolling and background-click deselection."""
        ui = self.gui.ui.imgui
        m = self.model
        if m is None:
            ui.text_disabled("Open an example to explore its model.")
            return
        ui.set_next_item_width(-1)
        changed, self.search = ui.input_text_with_hint("##model_search", "Search names or paths...", self.search)
        self._ensure_tree()
        tree = self.tree
        self.collision.sync()
        if changed or tree.query != self.search:
            tree.query = self.search
            tree.rebuild()
        if self.widgets.button("Collapse", small=True):
            tree.expanded.clear()
            tree.rebuild()
        ui.same_line()
        count = (
            sum(getattr(m, kind + "_count", 0) for kind in tree.kinds)
            + m.surface_count
            + m.volume_count
            + m.curve_count
            + m.deformable_visual_mesh_count
            + m.tendon_count
            + m.attachment_particle_body_count
            + m.constraint_mimic_count
            + m.actuator_count
        )
        count += sum(tree.custom.counts.values())
        worlds = "world" if m.world_count == 1 else "worlds"
        objects = "object" if count == 1 else "objects"
        ui.text_disabled(f"{m.world_count:,} {worlds} · {count:,} {objects}")
        ui.separator()
        scroll_height = ui.get_style().scrollbar_size + 2 if self._tree_content_width > self._tree_column_width else 0
        width_key = (id(tree), tree.revision, ui.get_font().font_id, ui.get_font_size(), self.widgets.scale)
        if width_key != self._tree_width_key:
            self._tree_content_width = 0.0
            self._tree_width_key = width_key
        # Scroll names independently so visibility and physics flags remain
        # reachable even in a deeply nested robot. Extents are measured only
        # for rows the clipper actually renders; no topology/device reads.
        height = max(
            1.0,
            ui.get_content_region_avail().y - scroll_height - (ui.get_style().item_spacing.y if scroll_height else 0),
        )
        if ui.begin_child("##tree_rows", ui.ImVec2(0, height)):
            table_flags = ui.TableFlags_.row_bg | ui.TableFlags_.sizing_stretch_prop
            if ui.begin_table("##scene_columns", 3, table_flags):
                ui.table_setup_column("Object", ui.TableColumnFlags_.width_stretch)
                ui.table_setup_column("View", ui.TableColumnFlags_.width_fixed, 25 * self.widgets.scale)
                ui.table_setup_column("Flags", ui.TableColumnFlags_.width_fixed, 50 * self.widgets.scale)
                ui.table_headers_row()
                clipper = ui.ListClipper()
                rows = tree.rows
                clipper.begin(len(rows))
                if self._reveal_selection:
                    selected_row = next((i for i, (key, _) in enumerate(rows) if key == self.selected), None)
                    if selected_row is not None:
                        clipper.include_item_by_index(selected_row)
                while clipper.step():
                    for row in range(clipper.display_start, clipper.display_end):
                        self._tree_row(*rows[row])
                clipper.end()
                ui.end_table()
            # Only the child background clears selection: row controls, the
            # search field and the scrollbar keep their own click behaviour.
            if (
                ui.is_window_hovered()
                and ui.is_mouse_clicked(0)
                and not ui.is_any_item_hovered()
                and not ui.is_any_item_active()
            ):
                self.select(None)
        ui.end_child()
        if scroll_height:
            ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(0, 0))
            ui.set_next_window_content_size(ui.ImVec2(self._tree_content_width, 1))
            if self._tree_scroll_pending is not None:
                ui.set_next_window_scroll(ui.ImVec2(self._tree_scroll_pending, 0))
                self._tree_scroll_pending = None
            ui.begin_child(
                "##tree_name_scroll",
                ui.ImVec2(max(1.0, self._tree_column_width), scroll_height),
                window_flags=ui.WindowFlags_.always_horizontal_scrollbar | ui.WindowFlags_.no_scroll_with_mouse,
            )
            self._tree_scroll_x = ui.get_scroll_x()
            ui.end_child()
            ui.pop_style_var()
        elif self._tree_content_width <= self._tree_column_width:
            self._tree_scroll_x = 0.0
            self._tree_scroll_pending = 0.0

    SHAPE_ICONS: ClassVar[dict[int, str]] = {
        int(GeoType.NONE): "empty",
        int(GeoType.MESH): "mesh",
        int(GeoType.CONVEX_MESH): "convex",
        int(GeoType.BOX): "box",
        int(GeoType.SPHERE): "sphere",
        int(GeoType.ELLIPSOID): "ellipsoid",
        int(GeoType.CAPSULE): "capsule",
        int(GeoType.CYLINDER): "cylinder",
        int(GeoType.PLANE): "plane",
        int(GeoType.CONE): "cone",
        int(GeoType.HFIELD): "terrain",
        int(GeoType.GAUSSIAN): "gaussian",
    }

    @staticmethod
    def joint_icon(joint_type: JointType | int) -> str:
        """Return the explorer icon name for a JointType value."""
        joint_enum = JointType(joint_type)
        return "joint_cable" if joint_enum == JointType.ROD else "joint_" + joint_enum.name.lower()

    def _tree_row(self, key: TreeKey, depth: int) -> None:
        """Draw a depth-indented explorer row with type icon, expansion, selection, visibility, and applicable flags."""
        ui, widgets, tree = self.gui.ui.imgui, self.widgets, self.tree
        kind, index = key
        resource = tree.resources.handles(key)
        if kind not in ("root", "group", "actuator_collection", "actuator_group", "battery") and not resource:
            tree.ensure_world(tree.world_for(key))
        ui.push_id(f"{kind}:{index}")
        ui.table_next_row()
        ui.table_next_column()
        self._tree_column_width = ui.get_content_region_avail().x
        name = tree.labels[key].rstrip("/").rsplit("/", 1)[-1]
        indent = depth * 10 * widgets.scale
        label_width = ui.get_tree_node_to_label_spacing() + ui.calc_text_size("        " + name).x + 8 * widgets.scale
        self._tree_content_width = max(self._tree_content_width, indent + label_width)
        if self._reveal_selection and self.selected == key:
            self._tree_scroll_x = max(
                0.0, min(indent, max(self._tree_scroll_x, indent + label_width - self._tree_column_width))
            )
            self._tree_scroll_pending = self._tree_scroll_x
        ui.set_cursor_pos_x(ui.get_cursor_pos_x() + indent - self._tree_scroll_x)
        pos = ui.get_cursor_screen_pos()
        flags = (
            ui.TreeNodeFlags_.no_tree_push_on_open
            | ui.TreeNodeFlags_.span_avail_width
            | ui.TreeNodeFlags_.open_on_arrow
            | ui.TreeNodeFlags_.open_on_double_click
            | ui.TreeNodeFlags_.frame_padding
        )
        has_children = tree.has_children(key) and not tree.query
        if not has_children:
            flags |= ui.TreeNodeFlags_.leaf
        if key in self.selection:
            flags |= ui.TreeNodeFlags_.selected
        ui.set_next_item_open(key in tree.expanded, ui.Cond_.always)
        if key in self.selection:
            ui.table_set_bg_color(ui.TableBgTarget_.row_bg0, widgets.color(theme.TREE_SELECTED))
            ui.push_style_color(ui.Col_.header, ui.ImVec4(*theme.TRANSPARENT))
            ui.push_style_color(ui.Col_.text, ui.ImVec4(*widgets.accent))
        ui.tree_node_ex("        " + name + "##node", flags)
        if key in self.selection:
            ui.pop_style_color(2)
        row_height = ui.get_item_rect_max().y - ui.get_item_rect_min().y
        if self._reveal_selection and self.selected == key:
            ui.set_scroll_here_y(0.5)
            self._reveal_selection = False
        if ui.is_item_clicked() and not ui.is_item_toggled_open():
            self.select(key, toggle=ui.get_io().key_ctrl, extend=ui.get_io().key_shift)
        if has_children and ui.is_item_toggled_open():
            tree.toggle(key)
        icon = {
            "root": "world",
            "group": "folder",
            "elements": "folder",
            "particle_set": "particles",
            "curve": "joint_cable",
            "deformable_visual_mesh": "mesh",
            "edge": "tri",
            "tendon_group": "folder",
            "tendon": "joint_cable",
            "tendon_guide": "joint_distance",
            "tendon_segment": "joint_cable",
            "attachment_group": "folder",
            "attachment_page": "folder",
            "attachment_particle_body": "joint_fixed",
            "mimic_group": "folder",
            "constraint_mimic": "link",
            "actuator_collection": "folder",
            "actuator_group": "custom",
            "battery": "custom",
        }.get(kind, kind)
        custom = (
            kind
            in (
                "frequency",
                "reference_group",
                "deformable_visual_mesh",
                "tendon_group",
                "tendon",
                "tendon_guide",
                "tendon_segment",
                "attachment_group",
                "attachment_page",
                "attachment_particle_body",
                "mimic_group",
                "constraint_mimic",
                "actuator_collection",
                "actuator_group",
                "battery",
            )
            or kind.startswith("attribute:")
            or tree.custom.frequency(key) is not None
            or self.collision.handles(key)
            or resource
        )
        if (
            kind in ("frequency", "reference_group")
            or kind.startswith("attribute:")
            or tree.custom.frequency(key) is not None
        ):
            icon = "folder" if kind in ("frequency", "reference_group") else "custom"
        if self.collision.handles(key):
            icon = "contact" if kind in ("contact", "soft_contact", "collision_contacts") else "collision"
        if resource:
            icon = (
                "bvh"
                if tree.resources.bvhs.handles(key)
                else "texture"
                if kind in ("sdf_texture", "shape_texture")
                else "sdf"
            )
        if kind in ("shape", "joint") and "type" not in tree.meta.get(key, {}):
            tree.meta.setdefault(key, {})["type"] = int(self.read(self.model, kind + "_type", index)[0])
        if kind == "shape":
            geo = tree.meta.get(key, {}).get("type", 0)
            icon = self.SHAPE_ICONS.get(geo, "shape")
        elif kind == "joint":
            joint_type = tree.meta.get(key, {}).get("type")
            if joint_type is not None:
                icon = self.joint_icon(joint_type)
        tint_kind = (
            icon
            if resource
            else "surface"
            if kind in ("curve", "deformable_visual_mesh")
            else "tri"
            if kind == "edge"
            else "joint"
            if kind in ("tendon", "tendon_guide", "tendon_segment", "attachment_particle_body", "constraint_mimic")
            else "custom"
            if custom
            else kind
        )
        icon_pos = (pos.x + ui.get_tree_node_to_label_spacing(), pos.y + (row_height - 16 * widgets.scale) / 2)
        widgets.icon(
            icon,
            icon_pos,
            16 * widgets.scale,
            active=key in self.selection,
            tint=widgets.explorer_colors.get(tint_kind),
            solid=True,
        )
        if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            icon_end = (icon_pos[0] + 16 * widgets.scale, icon_pos[1] + 16 * widgets.scale)
            if ui.is_mouse_hovering_rect(ui.ImVec2(*icon_pos), ui.ImVec2(*icon_end)):
                ui.set_tooltip(self._tree_type_name(key))
            else:
                full_name = tree.labels[key].rstrip("/")
                label_x = icon_pos[0] + ui.calc_text_size("        ").x
                clipped = label_x + ui.calc_text_size(name).x > ui.get_item_rect_max().x
                if clipped or full_name != name:
                    ui.set_tooltip(full_name)
        ui.table_next_column()
        if custom:
            ui.table_next_column()
            ui.pop_id()
            return
        if kind in ("surface", "volume", "particle_set", "elements", "particle", "tri", "edge", "tet"):
            if kind in ("surface", "volume", "particle_set"):
                visible = self.tree.deformables.visible(tree, key)
                if widgets.icon_button(
                    "##soft_visible",
                    "eye" if visible else "eye_off",
                    "Hide in viewport" if visible else "Show in viewport",
                    size=22,
                    flat=True,
                ):
                    if visible:
                        tree.deformables.hidden.add(key)
                    else:
                        tree.deformables.hidden.discard(key)
                    self._visibility_cache.clear()
            ui.table_next_column()
            ui.pop_id()
            return
        visible = self.object_visible(key)
        if widgets.icon_button(
            "##visible",
            "eye" if visible else "eye_off",
            "Hide in viewport" if visible else "Show in viewport",
            size=22,
            flat=True,
        ):
            self.toggle_visibility(key)
        ui.table_next_column()
        self._flag_buttons(key, compact=True)
        ui.pop_id()

    def _tree_type_name(self, key: TreeKey) -> str:
        """Describe the object represented by an icon, independently of its label."""
        kind, index = key
        if kind in ("shape", "joint"):
            value = self.tree.meta.get(key, {}).get("type")
            if value is None:
                value = int(self.read(self.model, kind + "_type", index)[0])
                self.tree.meta.setdefault(key, {})["type"] = value
            enum = GeoType if kind == "shape" else JointType
            try:
                label = f"{enum(value).name.replace('_', ' ').capitalize()} {kind}"
                if kind == "shape" and hasattr(self.tree, "resources"):
                    resources = self.tree.resources
                    source, sdf = resources.source(index), resources.shape_sdf(index)
                    if 0 <= sdf < resources.count:
                        label += f"\nModel SDF {sdf}"
                    if getattr(source, "sdf", None) is not None:
                        label += f"\nMesh SDF: {resources.sdf_storage(source.sdf)}"
                    if getattr(source, "texture", None) is not None:
                        label += "\nColor texture: 2D image"
                return label
            except ValueError:
                return f"{kind.capitalize()} type {value}"
        frequency = self.tree.custom.frequency(key)
        if frequency is not None:
            return f"Custom object: {frequency}"
        if kind == "frequency":
            return f"Custom frequency: {self.tree.custom.groups[key][0]}"
        if kind.startswith("attribute:"):
            return kind.split(":", 1)[1].replace("_", " ").capitalize() + " attributes"
        return {
            "root": "World" if kind == "root" and index >= 0 else "Shared world objects",
            "body": "Rigid body",
            "articulation": "Articulation",
            "group": "Object group",
            "elements": "Element group",
            "surface": "Triangle surface",
            "volume": "Tetrahedral volume",
            "curve": "Deformable curve / rod",
            "deformable_visual_mesh": "Skinned deformable visual mesh",
            "particle_set": "Particles",
            "particle": "Particle",
            "tri": "Triangle",
            "edge": "Bending edge",
            "tet": "Tetrahedron",
            "tendon_group": "Massless routed tendons",
            "tendon": "Massless routed tendon",
            "tendon_guide": "Tendon guide point",
            "tendon_segment": "Tendon segment",
            "attachment_group": "Particle-body attachments",
            "attachment_page": "Particle-body attachments",
            "attachment_particle_body": "Particle-body attachment",
            "mimic_group": "Joint mimic constraints",
            "constraint_mimic": "Joint mimic constraint",
            "actuator_collection": "Actuators and batteries",
            "actuator_group": "Actuator group",
            "battery": "Battery group",
            "contact": "Rigid contact",
            "soft_contact": "Soft-rigid contact",
            "collision_pair": "Candidate shape pair",
            "collision_pairs": "Candidate shape pairs",
            "collision": "Collision diagnostics",
            "collision_contacts": "Contacts",
            "collision_pipeline": "Collision pipeline",
            "collision_broad": "Broadphase",
            "collision_narrow": "Narrowphase",
            "collision_page": "Collision results",
            "collision_group": "Contact group",
            "reference_group": "Referenced attributes",
            "sdf_collection": "Shared Model signed-distance fields",
            "sdf_page": "Signed-distance fields",
            "sdf": "Collision signed-distance field",
            "shape_sdf": "Reference to a shared Model SDF",
            "mesh_sdf": "Signed-distance field stored on the source mesh",
            "sdf_texture": "3D texture containing signed-distance samples",
            "sdf_slots": "Coarse cells mapped to packed narrow-band subgrids",
            "shape_texture": "2D surface-color texture",
            "bvh_collection": "Bounding-volume hierarchies",
            "bvh": "Query or collision BVH",
            "bvh_mesh": "Triangle-mesh BVH shared by shape instances",
            "bvh_page": "Mesh BVHs",
            "shape_bvh": "BVH for this shape's geometry",
            "bvh_gaussian": "Gaussian BVH",
            "bvh_optix": "OptiX shape acceleration structure",
        }.get(kind, kind.replace("_", " ").capitalize())

    def object_visible(self, key: TreeKey) -> bool:
        """Resolve effective visibility from world visibility and cached visible shape or deformable descendants."""
        world = self.tree.world_for(key)
        if key[0] == "root" and world >= 0:
            return self.viewer._should_render_world(world)
        if world >= 0 and not self.viewer._should_render_world(world):
            return False
        if key not in self._visibility_cache:
            self._visibility_cache[key] = any(
                i not in self._hidden_shapes and self.tree.meta.get(("shape", i), {}).get("opacity", 1) > 0
                for i in self.tree.shape_descendants(key)
            ) or any(item not in self.tree.deformables.hidden for item in self.tree.deformable_descendants(key))
        return self._visibility_cache[key]

    def toggle_visibility(self, key: TreeKey) -> None:
        """Toggle a world or subtree while preserving world offsets and previous shape opacities.

        Update shape opacity on the device in one batch and read previous values
        back only when hiding. Invalidate visibility caches and renderer batches
        when their inputs change.
        """
        world = self.tree.world_for(key)
        if world >= 0 and (key[0] == "root" or not self.viewer._should_render_world(world)):
            visible = (
                set(range(self.model.world_count))
                if self.viewer._visible_worlds is None
                else set(self.viewer._visible_worlds)
            )
            if key[0] == "root" and world in visible:
                visible.remove(world)
            else:
                visible.add(world)
            self.viewer.set_visible_worlds(
                None if len(visible) == self.model.world_count else sorted(visible), reflow=False
            )
            self._visibility_cache.clear()
            return
        shapes = self.tree.shape_descendants(key)
        hide = self.object_visible(key)
        deformables = self.tree.deformable_descendants(key)
        if hide:
            self.tree.deformables.hidden.update(deformables)
        else:
            self.tree.deformables.hidden.difference_update(deformables)
        self._visibility_cache.clear()
        indices = [i for i in shapes if not hide or i not in self._hidden_shapes]
        if not indices:
            return
        values = [0.0 if hide else self._hidden_shapes.get(i, 1.0) for i in indices]
        device = self.model.device
        previous = wp.empty(len(indices), dtype=float, device=device)
        wp.launch(
            _set_shape_opacity,
            dim=len(indices),
            inputs=[
                self.model.shape_opacity,
                wp.array(indices, dtype=int, device=device),
                wp.array(values, dtype=float, device=device),
                previous,
            ],
            device=device,
        )
        # One transfer per interaction, including for a whole subtree.
        if hide:
            self._hidden_shapes.update(zip(indices, previous.numpy().tolist(), strict=True))
        else:
            for i, value in zip(indices, values, strict=True):
                self._hidden_shapes.pop(i, None)
                self.tree.meta.setdefault(("shape", i), {})["opacity"] = value
        self._visibility_cache.clear()
        self.viewer.model_changed = True
        self._cache.clear()

    def _flag_buttons(self, key: TreeKey, *, compact: bool = False) -> None:
        """Draw body or shape flags with distinct active and read-only states and route permitted edits."""
        ui, widgets = self.gui.ui.imgui, self.widgets
        kind, index = key
        flags = int(self.tree.meta.get(key, {}).get("flags", 0))
        if kind == "body":
            runtime, readonly_reason = self._body_flag_support()
            enabled = runtime and self.can_edit_state() and not flags & int(BodyFlags.PROXY)
            options = (
                (BodyFlags.DYNAMIC, "dynamic", "Dynamic: responds to forces"),
                (BodyFlags.KINEMATIC, "kinematic", "Kinematic: prescribed motion"),
            )
        elif kind == "shape":
            enabled = False
            readonly_reason = "Read only: configured when the model and collision pipeline are built."
            options = (
                (ShapeFlags.COLLIDE_SHAPES, "collision", "Shape collisions"),
                (ShapeFlags.COLLIDE_PARTICLES, "particles", "Particle collisions"),
            )
            if not compact:
                options += (
                    (ShapeFlags.VISIBLE, "eye", "Visible geometry"),
                    (ShapeFlags.SITE, "site", "Reference site"),
                    (ShapeFlags.HYDROELASTIC, "hydro", "Hydroelastic contact"),
                )
        else:
            return
        for i, (flag, icon, label) in enumerate(options):
            if i:
                ui.same_line(spacing=1 * widgets.scale)
            editable = enabled or (kind == "shape" and flag == ShapeFlags.VISIBLE)
            tooltip = label + (" (on)" if flags & int(flag) else " (off)")
            if not editable:
                tooltip += (
                    "\nRead only: proxy body."
                    if kind == "body" and flags & int(BodyFlags.PROXY)
                    else "\nPause at the live frame to edit."
                    if kind == "body" and runtime
                    else f"\n{readonly_reason}"
                )
            if widgets.icon_button(
                f"##flag{int(flag)}",
                icon,
                tooltip,
                active=bool(flags & int(flag)),
                enabled=editable,
                size=22,
                flat=compact,
            ):
                self.set_flag(kind, index, int(flag))

    def _at_live_frame(self) -> bool:
        """Report whether the displayed timeline is at its latest frame rather than a historical snapshot."""
        history = getattr(self, "history", None)
        return history is None or not history.replaying

    def logged_observables(self) -> SolverBase.Observables | None:
        """Return solver observables explicitly supplied via ``log_observables()`` or ``log_contacts()``."""
        if self.model is None:
            return None
        for attr in ("_last_observables", "_last_contact_observables"):
            obs = getattr(self.viewer, attr, None)
            if obs is not None and obs.model is self.model:
                return obs
        return None

    def _active_world(self) -> int:
        """Resolve the world index matching the current selection or explorer root."""
        world = getattr(self, "world", 0)
        tree = getattr(self, "tree", None)
        if tree is not None:
            if self.selected is not None:
                world = tree.world_for(self.selected)
            elif ("root", world) not in tree.roots and ("root", -1) in tree.roots:
                world = -1
        return world

    def _active_world_slot(self) -> int:
        """Map the active world (including global world ``-1``) to a ``world_count + 1`` array index."""
        if self.model is None:
            return 0
        world = self._active_world()
        return self.model.world_count if world < 0 else min(world, self.model.world_count)

    def _read_world_column(self, array: wp.array, slot: int) -> np.ndarray:
        """Read one world column of a 2D ``(steps, world_count + 1)`` observable array once per frame."""
        key = (id(array), "world_col", slot)
        if key not in self._live_cache:
            host = self._live_cache.get((id(array), "host_2d"))
            if host is None:
                host = np.asarray(array.numpy())
                self._live_cache[(id(array), "host_2d")] = host
            self._live_cache[key] = np.asarray(host[:, slot]) if slot < host.shape[1] else np.empty(0, dtype=host.dtype)
        return self._live_cache[key]

    def can_edit_state(self) -> bool:
        """Allow state mutation only while paused at the newest retained frame."""
        return self.viewer.is_paused() and self._at_live_frame()

    def can_edit_control(self) -> bool:
        """Allow live inputs while running or paused, but never during historical replay."""
        return self._at_live_frame()

    def _capture_control_initial(self) -> None:
        """Keep one device-side baseline per example's Control, before GUI edits."""
        # History temporarily substitutes display_control during rendering.
        # Its values must never replace the live example's reset baseline.
        if not self._at_live_frame():
            return
        control = self.bindings.control
        if control is not getattr(self, "_control_owner", None):
            self._control_owner = control
            self._control_initial = {
                name: wp.clone(array, requires_grad=False) for name, array in _component_arrays(control).items()
            }

    def reset_control(self, *, zero: bool = False) -> bool:
        """Reset the complete live Control in place, preserving CUDA graph bindings."""
        if not self.can_edit_control():
            self.message = "Return to the live frame to reset Control inputs."
            return False
        self._capture_control_initial()
        arrays = _component_arrays(self.bindings.control)
        if not arrays:
            return False
        initial = self._control_initial
        # Check the complete operation before mutating any input. A dynamically
        # added field gets its baseline when first encountered, before editing.
        for name, array in arrays.items():
            if name not in initial:
                initial[name] = wp.clone(array, requires_grad=False)
            elif array.shape != initial[name].shape or array.dtype != initial[name].dtype:
                self.message = "Control layout changed. Restart the example to capture new initial values."
                return False
        for name, array in arrays.items():
            if zero:
                array.zero_()
            else:
                wp.copy(array, initial[name])
        # Match Control.clear(model)'s position-target exception. Quaternion
        # slots must remain valid; restore last in case a custom array aliases
        # the position targets, including Control that shares Model storage.
        if zero and "joint_target_q" in arrays:
            wp.copy(arrays["joint_target_q"], initial["joint_target_q"])
        self._undo = self._drag_undo = None
        self._live_cache.clear()
        self._drafts.clear()
        if self.history is not None and self.viewer.is_paused():
            self.history.replace_next_frame()
        self.message = ""
        return True

    def is_control_owner(self, owner: object) -> bool:
        """Recognize the live Control and its registered namespace containers."""
        return self.bindings.owns_control(owner)

    def _body_flag_support(self) -> tuple[bool, str]:
        """Return whether the bound solver supports runtime body-motion edits and a read-only reason if not."""
        solver = self.bindings.solver
        if not isinstance(solver, SolverBase) or self.model is None:
            return False, "Read only: configured when the model and collision pipeline are built."
        if type(solver).__name__ in ("SolverImplicitMPM", "SolverKamino", "SolverStyle3D"):
            return False, "Read only: configured when the model and solver are built."
        if self.model.device.is_cuda and not solver.is_notify_model_changed_graphable(ModelFlags.BODY_PROPERTIES):
            return False, "Read only: changing body motion on CUDA requires recapturing the solver step."
        return True, ""

    def set_flag(self, kind: str, index: int, flag: int) -> None:
        """Set a body motion flag or toggle shape visibility, notifying its solver or renderer.

        Reject unsupported edits and proxy-body changes. Restore the original
        array value and solver caches if notification fails.
        """
        if self.model is None:
            return
        array = getattr(self.model, kind + "_flags")
        previous = int(array[index : index + 1].numpy()[0])
        value = flag if kind == "body" else previous ^ flag
        if kind == "body":
            runtime, reason = self._body_flag_support()
            if not runtime:
                self.message = reason
                return
            if not self.can_edit_state():
                self.message = "Pause at the live frame to edit."
                return
            if previous & int(BodyFlags.PROXY) or value == previous:
                return
        elif kind != "shape" or flag != int(ShapeFlags.VISIBLE):
            return
        array[index : index + 1].fill_(value)
        if kind == "body":
            solver = self.bindings.solver
            try:
                solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
            except Exception as exc:
                array[index : index + 1].fill_(previous)
                rollback_exc: Exception | None = None
                try:
                    solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
                except Exception as recovery_exc:
                    rollback_exc = recovery_exc
                self._cache.clear()
                self._live_cache.clear()
                if rollback_exc is not None:
                    self.message = (
                        f"Solver rejected body flag change ({exc}) and rollback failed ({rollback_exc}); "
                        "restart the simulation to resynchronize solver state."
                    )
                else:
                    self.message = f"Solver rejected body flag change: {exc}"
                return
            if self.history is not None and self.viewer.is_paused():
                self.history.replace_next_frame()
            self.message = ""
        else:
            try:
                self.viewer._rebuild_shape_batches_for_opacity_groups()
                self.gui.update_shape_counts(self.model)
            except Exception:
                array[index : index + 1].fill_(previous)
                raise
        self.tree.meta[(kind, index)]["flags"] = value
        self._cache.clear()
        self._live_cache.clear()

    _BODY_VARIANT_FLAGS: ClassVar[int] = int(
        ModelFlags.SHAPE_TYPE_AND_SOURCE | ModelFlags.SHAPE_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES
    )

    def _body_variant_support(self) -> tuple[bool, str]:
        """Return whether the bound solver supports runtime body-variant edits and a read-only reason if not."""
        solver = self.bindings.solver
        if not isinstance(solver, SolverBase) or self.model is None:
            return False, "Read only: bind a solver with viewer.set_solver() to switch body variants."
        if type(solver).__name__ in ("SolverImplicitMPM", "SolverStyle3D"):
            return False, "Read only: configured when the model and solver are built."
        if self.model.device.is_cuda and (
            (
                type(solver).__name__ != "SolverSteel"
                and not solver.is_notify_model_changed_graphable(self._BODY_VARIANT_FLAGS)
            )
            or (
                type(solver).__name__ == "SolverFeatherstone" and getattr(solver, "update_mass_matrix_interval", 1) != 1
            )
        ):
            return False, "Read only: switching body variants on CUDA requires recapturing the solver step."
        return True, ""

    @staticmethod
    def _snapshot_body_variant_state(model: Model, body: int) -> list[tuple[wp.array, int, int, np.ndarray]]:
        """Snapshot the live Model slices touched by ``model.select_body_variants()`` for one body."""
        saved: list[tuple[wp.array, int, int, np.ndarray]] = []
        seen: set[int] = set()

        def capture(array: Any, start: int, stop: int) -> None:
            if not isinstance(array, wp.array) or stop <= start or id(array) in seen:
                return
            seen.add(id(array))
            saved.append((array, start, stop, array[start:stop].numpy().copy()))

        for name in ("body_variant", "body_mass", "body_inv_mass", "body_com", "body_inertia", "body_inv_inertia"):
            capture(getattr(model, name, None), body, body + 1)
        if model._body_variant_rows is not None and body < len(model._body_variant_rows):
            row_start, row_count = (int(v) for v in model._body_variant_rows[body : body + 1].numpy()[0])
            row_stop = row_start + row_count
            if row_count > 0:
                for name in ("_shape_variant_shape", "shape_flags", "bvh_shape_bounds", *model._variant_shape_geometry):
                    capture(getattr(model, name, None), row_start, row_stop)
                for _variant_values, target in model._variant_shape_custom.values():
                    capture(target, row_start, row_stop)
        return saved

    def select_body_variant(self, body: int, variant: int) -> bool:
        """Switch one body's active shape variant while paused at the live frame.

        Require a bound solver so simulation caches stay synchronized. Restore
        the exact live pre-switch Model slices and refresh solver caches if
        solver notification fails.
        """
        supported, reason = self._body_variant_support()
        if not supported:
            self.message = reason
            return False
        if not self.can_edit_state():
            self.message = "Pause at the live frame to switch body variants."
            return False
        model = self.model
        if model is None or model.body_variant is None:
            return False
        labels = model.body_variant_labels(body)
        if not (0 <= variant < len(labels)):
            return False
        previous = int(self.read(model, "body_variant", body)[0])
        if variant == previous:
            return True
        saved = self._snapshot_body_variant_state(model, body)
        device = model.device
        bodies_arr = wp.array([body], dtype=wp.int32, device=device)
        flags = model.select_body_variants(wp.array([variant], dtype=wp.int32, device=device), bodies=bodies_arr)
        solver = self.bindings.solver
        try:
            solver.notify_model_changed(flags)
        except Exception as exc:
            for array, start, stop, values in saved:
                array[start:stop].assign(values)
            rollback_exc: Exception | None = None
            try:
                solver.notify_model_changed(flags)
            except Exception as recovery_exc:
                rollback_exc = recovery_exc
            self.viewer.notify_model_changed(flags)
            self._cache.clear()
            self._live_cache.clear()
            if rollback_exc is not None:
                self.message = (
                    f"Solver rejected variant change ({exc}) and rollback failed ({rollback_exc}); "
                    "restart the simulation to resynchronize solver state."
                )
            else:
                self.message = f"Solver rejected variant change: {exc}"
            return False
        self.viewer.notify_model_changed(flags)
        if self.history is not None and self.viewer.is_paused():
            self.history.replace_next_frame()
        self._cache.clear()
        self._live_cache.clear()
        self.message = ""
        return True

    def _inspector(self) -> None:
        """Draw property, environment, solver, and available example tabs with history edit guards."""
        ui = self.gui.ui.imgui
        if ui.begin_tab_bar("##inspector_tabs"):
            if self.widgets.tab_item(
                "Properties", flags=ui.TabItemFlags_.set_selected if self._focus_properties else 0
            )[0]:
                self._focus_properties = False
                self._properties()
                ui.end_tab_item()
            if self.widgets.tab_item(
                "Environment", flags=ui.TabItemFlags_.set_selected if self._focus_environment else 0
            )[0]:
                self._focus_environment = False
                self._environment()
                ui.end_tab_item()
            if self.widgets.tab_item("Solver")[0]:
                self._solver_properties()
                ui.end_tab_item()
            callbacks = self._custom_callbacks()
            label = "Controls"
            focus = bool(callbacks) and not self._custom_tab_shown
            self._custom_tab_shown |= focus
            flags = ui.TabItemFlags_.set_selected if focus else 0
            if callbacks and self.widgets.tab_item(label, accent=True, flags=flags)[0]:
                ui.begin_disabled(self.history is not None and self.history.replaying)
                for callback in callbacks:
                    callback(ui)
                ui.end_disabled()
                ui.end_tab_item()
            ui.end_tab_bar()

    def _custom_callbacks(self) -> list[Callable[[Any], None]]:
        """Return ``"panel"`` then ``"side"`` UI callbacks for the inspector's Controls tab.

        The viewer's Layers header has nothing to show until a second layer exists.
        """
        panels = [
            callback
            for callback in self.gui._ui_callbacks["panel"]
            if not (callback == self.viewer._ui_populate_layers_panel and len(self.viewer.layers) <= 1)
        ]
        return panels + self.gui._ui_callbacks["side"]

    def _environment(self) -> None:
        """Keep scene forces accessible independently of the current selection."""
        w, m = self.widgets, self.model
        if m is None:
            return
        with w.group("Environment") as opened:
            if opened:
                w.axis("Up axis", int(m.up_axis))
                world = self._active_world()
                slot = len(m.gravity) - 1 if world < 0 else min(world, len(m.gravity) - 1)
                gravity = self.read(m, "gravity", slot)
                if gravity.size:
                    gravity_label = (
                        "Gravity (Shared)"
                        if world < 0 and m.world_count > 1
                        else f"Gravity (World {world})"
                        if m.world_count > 1
                        else "Gravity"
                    )
                    w.vec3(gravity_label, gravity, unit="m/s²")
        wind = getattr(self.viewer, "wind", None)
        if wind is None or not m.particle_count:
            return
        editable = self._at_live_frame()
        reason = "Return to the live frame to edit wind."
        with w.group("Wind") as opened:
            if opened:
                for label, name, speed, bounds, unit in (
                    ("Amplitude", "amplitude", 0.01, (0, 1000), "m/s²"),
                    ("Period", "period", 0.05, (0.01, 1000), "s"),
                    ("Frequency", "frequency", 0.01, (0, 100), "×"),  # noqa: RUF001 - dimensionless multiplier
                ):
                    edit = w.number(
                        label,
                        getattr(wind, name),
                        editable=editable,
                        speed=speed,
                        bounds=bounds,
                        unit=unit,
                        reason=reason,
                    )
                    if edit.changed and math.isfinite(edit.value):
                        setattr(wind, name, edit.value)
                edit = w.vec3("Direction", wind.direction, editable=editable, speed=0.01, reason=reason)
                if edit.changed and np.isfinite(edit.value).all():
                    wind.direction = edit.value.tolist()

    def _breadcrumbs(self) -> None:
        """Draw a width-bounded ancestor path and optional selection-follow toggle."""
        ui, tree, scale = self.gui.ui.imgui, self.tree, self.widgets.scale
        chain, seen, key = [], set(), self.selected
        while key is not None and key not in seen:
            seen.add(key)
            chain.append(key)
            key = tree.parents.get(key)
        chain.reverse()
        available = ui.get_content_region_avail().x
        follow_button = self.can_follow_selection
        button_size = ui.get_text_line_height()
        button_x = ui.get_cursor_pos_x() + available - button_size
        row_y = ui.get_cursor_pos_y()
        if follow_button:
            available -= button_size + 8 * scale
        gap = 18 * scale
        max_label = min(120 * scale, max(30 * scale, (available - 40 * scale - 2 * gap) / 2))

        def label(key: TreeKey) -> str:
            name = tree.labels[key].rstrip("/").rsplit("/", 1)[-1]
            if ui.calc_text_size(name).x <= max_label:
                return name
            low, high = 0, len(name)
            while low < high:
                middle = (low + high + 1) // 2
                if ui.calc_text_size(name[:middle] + "…").x <= max_label:
                    low = middle
                else:
                    high = middle - 1
            return name[:low] + "…"

        def navigate(key: TreeKey) -> None:
            self.navigate(key)

        labels = [label(key) for key in chain]
        width = sum(ui.calc_text_size(name).x + 2 * ui.get_style().frame_padding.x for name in labels)
        overflow = len(chain) > 2 and width + gap * (len(chain) - 1) > available
        ui.push_id("##property_path")
        if overflow:
            if self.widgets.button("…##ancestors", small=True):
                ui.open_popup("##ancestor_menu")
            if ui.is_item_hovered():
                ui.set_tooltip("Navigate to an ancestor")
            if ui.begin_popup("##ancestor_menu"):
                for ancestor in chain[:-2]:
                    name = tree.labels[ancestor].rstrip("/").rsplit("/", 1)[-1]
                    if ui.selectable(f"{name}##{ancestor}", False)[0]:
                        navigate(ancestor)
                ui.end_popup()
            chain, labels = chain[-2:], labels[-2:]
        for i, (key, name) in enumerate(zip(chain, labels, strict=True)):
            if i or overflow:
                ui.same_line(spacing=4 * scale)
                ui.text_disabled("›")  # noqa: RUF001 - breadcrumb chevron
                ui.same_line(spacing=4 * scale)
            current = key == self.selected
            ui.push_style_color(
                ui.Col_.button, ui.ImVec4(*theme.BREADCRUMB_SELECTED if current else theme.BREADCRUMB_BACKGROUND)
            )
            if self.widgets.button(f"{name}##crumb:{key[0]}:{key[1]}", small=True) and not current:
                navigate(key)
            ui.pop_style_color()
            if ui.is_item_hovered(ui.HoveredFlags_.delay_normal) and name.endswith("…"):
                ui.set_tooltip(tree.labels[key].rstrip("/").rsplit("/", 1)[-1])
        ui.pop_id()
        if follow_button:
            ui.same_line()
            ui.set_cursor_pos(ui.ImVec2(button_x, row_y))
            if self.widgets.icon_button(
                "##follow_selection",
                "follow",
                "Stop following selection" if self.gui._follow_latched else "Follow selection (hold F)",
                active=self.gui.following_selection,
                size=button_size / scale,
            ):
                self.gui.toggle_follow_selection()
        ui.spacing()

    def _properties(self) -> None:
        """Dispatch selection properties to collision, resource, or generic attribute inspectors."""
        ui, w, m = self.gui.ui.imgui, self.widgets, self.model
        if m is None:
            return
        self._ensure_tree()
        if len(self.selection) > 1:
            with w.group(f"Selection · {len(self.selection):,}") as opened:
                if opened:
                    for kind in sorted({key[0] for key in self.selection}):
                        w.number(
                            kind.replace("_", " ").title(), sum(key[0] == kind for key in self.selection), integer=True
                        )
                    if w.icon_button("##frame_selection", "focus", "Frame selection (F)"):
                        self.gui.frame_camera_on_model()
        active_only = self.collision.handles(self.selected) or self.tree.resources.handles(self.selected)
        if len(self.selection) > 1 and active_only:
            ui.text_disabled("Active item")
        if self.selected is not None and (len(self.selection) <= 1 or active_only):
            self._breadcrumbs()
        if len(self.selection) == 1 and self.selected and self.selected[0] in ("body", "shape"):
            self._flag_buttons(self.selected)
        if not hasattr(self, "property_inspector"):
            from .properties import PropertyInspector  # noqa: PLC0415

            self.property_inspector = PropertyInspector(self)
        if self.collision.handles(self.selected):
            self.collision.draw(self.property_inspector)
            return
        if self.tree.resources.handles(self.selected):
            self.collision.contacts_tab = False
            self.tree.resources.draw(self)
            return
        self.property_inspector.draw()
        if self.message:
            ui.separator()
            ui.text_wrapped(self.message)

    @classmethod
    def _uses_maximal_coordinates(cls, solver: object | None) -> bool:
        """Report whether a solver steps maximal body coordinates without updating reduced joint_q/joint_qd."""
        name = type(solver).__name__
        if name in ("SolverXPBD", "SolverVBD", "SolverSemiImplicit", "SolverKamino"):
            return True
        if isinstance(entries := getattr(solver, "_entries", None), dict):
            return any(cls._uses_maximal_coordinates(getattr(entry, "solver", None)) for entry in entries.values())
        return False

    def _joint_state_support(self, solver: object | None = ...) -> tuple[bool, str]:
        """Return whether the bound solver synchronizes runtime joint-state edits on resume."""
        if solver is ...:
            solver = self.bindings.solver
        if solver is None:
            return True, ""
        name = type(solver).__name__
        if name in ("SolverImplicitMPM", "SolverStyle3D"):
            return False, "Read only: this solver does not step articulated joint coordinates."
        if name == "SolverMuJoCo":
            if getattr(solver, "update_data_interval", 1) != 1:
                return (
                    False,
                    "Read only: SolverMuJoCo with update_data_interval != 1 does not synchronize joint state every step.",
                )
            if getattr(solver, "enable_sleeping", False):
                return False, "Read only: SolverMuJoCo sleeping caches derived tree state across steps."
        if isinstance(entries := getattr(solver, "_entries", None), dict):
            for entry in entries.values():
                supported, reason = self._joint_state_support(getattr(entry, "solver", None))
                if not supported:
                    return False, reason
        return True, ""

    def _solver_properties(self) -> None:
        """Show generic ``SolverBase`` metadata and logged solver observables."""
        ui, w = self.gui.ui.imgui, self.widgets
        solver = self.bindings.solver
        observables = self.logged_observables()
        if solver is None and observables is None:
            ui.text_wrapped("No solver or observables supplied. Call viewer.set_solver() or viewer.log_observables().")
            return
        if solver is not None:
            ui.text(type(solver).__name__.removeprefix("Solver"))
            ui.spacing()
            if (
                isinstance(solver, SolverBase)
                and solver.supports_collision_pipeline
                and solver.collision_pipeline is not None
            ):
                freq_types = solver.collision_frequency_type
                freqs = solver.collision_frequency
                reason = "Collision schedule configured on SolverBase."
                with w.group("Collision schedule") as opened:
                    if opened:
                        for slot in SolverBase.CollisionSlot:
                            ftype = freq_types.get(slot)
                            if ftype is None:
                                continue
                            label = slot.name.replace("_", " ").capitalize()
                            text = ftype.name.replace("_", " ").title()
                            if ftype == SolverBase.CollisionFrequencyType.ITERATIONS:
                                text = f"{text} (every {freqs.get(slot, 1)})"
                            w.text_value(label, text, reason=reason)
        self._solver_observables(solver, observables)

    def _solver_observables(self, solver: Any, observables: SolverBase.Observables | None) -> None:
        """Render logged solver observables through ``PropertyInspector`` and list unrequested capabilities."""
        if self.model is None:
            return
        w = self.widgets
        Frequency, Assignment = Model.AttributeFrequency, Model.AttributeAssignment
        supported = solver.supported_observables if isinstance(solver, SolverBase) else frozenset()
        logged_kinds: set[str] = set()
        entries_1d: list[tuple[AttributeRows, str, Model.AttributeSpec, object, str, Any]] = []
        entries_2d: list[tuple[AttributeRows, str, Model.AttributeSpec, wp.array]] = []
        empty_labels: list[tuple[str, str]] = []
        if observables is not None:
            for kind, (attr_name, field_spec) in observables._observable_fields().items():
                if kind not in observables.kinds:
                    continue
                array = getattr(observables, attr_name, None)
                if not isinstance(array, wp.array):
                    continue
                logged_kinds.add(kind)
                freq = field_spec.frequency
                spec = Model.AttributeSpec(freq)
                if array.ndim == 2 and freq == Frequency.WORLD:
                    if array.size == 0 or array.shape[1] == 0:
                        empty_labels.append((attr_name, "Empty"))
                    else:
                        slot = min(self._active_world_slot(), array.shape[1] - 1)
                        entries_2d.append((AttributeRows(self.model, Frequency.WORLD, [slot]), attr_name, spec, array))
                elif freq in (Frequency.CONTACT, Frequency.CONTACT_RIGID, Frequency.CONTACT_SOFT):
                    contacts = observables.contacts_for_kind(kind) or self.collision.contacts
                    if contacts is None:
                        empty_labels.append((attr_name, "No contacts"))
                        continue
                    counters = self.read(contacts, "contact_counters", 0, 2, live=True)
                    n_rigid = (
                        max(0, min(int(counters[0]), contacts.rigid_contact_max, len(array)))
                        if len(counters) > 0
                        else 0
                    )
                    n_soft = max(0, min(int(counters[1]), contacts.soft_contact_max)) if len(counters) > 1 else 0
                    if freq == Frequency.CONTACT_RIGID:
                        indices: Any = range(n_rigid)
                    elif freq == Frequency.CONTACT_SOFT:
                        indices = range(min(n_soft, len(array)))
                    elif n_soft == 0:
                        indices = range(n_rigid)
                    else:
                        soft_end = min(contacts.rigid_contact_max + n_soft, len(array))
                        indices = np.r_[
                            np.arange(n_rigid, dtype=np.int32),
                            np.arange(contacts.rigid_contact_max, soft_end, dtype=np.int32),
                        ]
                    if len(indices) == 0:
                        empty_labels.append((attr_name, "0 active contacts"))
                    else:
                        entries_1d.append(
                            (AttributeRows(self.model, freq, indices), attr_name, spec, observables, attr_name, array)
                        )
                elif freq == Frequency.WORLD:
                    if len(array) == 0:
                        empty_labels.append((attr_name, "Empty"))
                    elif self.model.world_count <= 1:
                        slot = min(self._active_world_slot(), len(array) - 1)
                        entries_1d.append(
                            (
                                AttributeRows(self.model, Frequency.WORLD, [slot]),
                                attr_name,
                                spec,
                                observables,
                                attr_name,
                                array,
                            )
                        )
                    else:
                        world_indices = list(range(min(self.model.world_count, len(array))))
                        if (
                            self.tree is not None
                            and ("root", -1) in self.tree.roots
                            and self.model.world_count < len(array)
                        ):
                            world_indices.append(self.model.world_count)
                        entries_1d.append(
                            (
                                AttributeRows(self.model, Frequency.WORLD, world_indices),
                                attr_name,
                                spec,
                                observables,
                                attr_name,
                                array,
                            )
                        )
                elif freq == Frequency.ONCE:
                    if len(array) == 0:
                        empty_labels.append((attr_name, "Empty"))
                    else:
                        entries_1d.append(
                            (
                                AttributeRows(self.model, Frequency.ONCE, [0]),
                                attr_name,
                                spec,
                                observables,
                                attr_name,
                                array,
                            )
                        )
                elif freq in (Frequency.JOINT_DOF, Frequency.JOINT_COORD):
                    if self.model.joint_count == 0 or len(array) == 0:
                        empty_labels.append((attr_name, "0 active DOFs"))
                    else:
                        offsets = "joint_q_start" if freq == Frequency.JOINT_COORD else "joint_qd_start"
                        spans = self.read(self.model, offsets, 0, self.model.joint_count + 1)
                        active_joints = np.flatnonzero(spans[1:] > spans[:-1])
                        if len(active_joints) == 0:
                            empty_labels.append((attr_name, "0 active DOFs"))
                        else:
                            entries_1d.append(
                                (
                                    AttributeRows(self.model, freq, active_joints, joints=True),
                                    attr_name,
                                    spec,
                                    observables,
                                    attr_name,
                                    array,
                                )
                            )
                else:
                    if len(array) == 0:
                        empty_labels.append((attr_name, "Empty"))
                    else:
                        entries_1d.append(
                            (
                                AttributeRows(self.model, freq, range(len(array))),
                                attr_name,
                                spec,
                                observables,
                                attr_name,
                                array,
                            )
                        )

        if not supported and not logged_kinds:
            return
        obs_reason = "Computed by the solver during step(); read-only."
        unreq_reason = "Request with solver.observables() and pass to viewer.log_observables() or log_contacts()."
        with w.group("Observables", default_open=bool(logged_kinds)) as opened:
            if not opened:
                return
            for entry in entries_1d:
                count = len(entry[0].indices)
                default_open = count <= 2 or (count <= 4 and getattr(entry[5].dtype, "_length_", 1) <= 3)
                self.property_inspector._fields([entry], Assignment.STATE, flat=True, default_open=default_open)
            if entries_2d:
                self.property_inspector._draw_observables_2d(entries_2d)
            for label, status in empty_labels:
                w.text_value(label, status, reason=obs_reason)
            unrequested = sorted(supported - logged_kinds)
            if not logged_kinds:
                for kind in unrequested:
                    w.text_value(kind.replace("_", " ").capitalize(), "Available", reason=unreq_reason)
            elif unrequested:
                with w.disclosure(f"Available ({len(unrequested)} not requested)", default_open=False) as show_unreq:
                    if show_unreq:
                        for kind in unrequested:
                            w.text_value(kind.replace("_", " ").capitalize(), "Available", reason=unreq_reason)

    def live_state(self) -> State | None:
        """Resolve the application live State through the bindings, falling back to the last logged State."""
        return self.viewer._last_state

    def joint_display_state(self) -> State | SimpleNamespace | None:
        """Recover reduced coordinates for maximal-coordinate solvers on demand."""
        state = self.live_state()
        shown = getattr(self.viewer, "_last_state", None)
        if state is None:
            return shown
        historical = not self._at_live_frame()
        maximal = self._uses_maximal_coordinates(self.bindings.solver)
        if not maximal and not historical:
            return state
        articulations = set()
        for kind, index in getattr(self, "selection", ()) or ((self.selected,) if self.selected else ()):
            if kind == "articulation":
                articulations.add(index)
            elif kind == "joint":
                art = int(self.read(self.model, "joint_articulation", index)[0])
                if art >= 0:
                    articulations.add(art)
        if not articulations:
            return shown or state
        selected = tuple(sorted(articulations))
        if self._kinematic_state is None:
            self._kinematic_state = SimpleNamespace(
                joint_q=wp.empty_like(state.joint_q), joint_qd=wp.empty_like(state.joint_qd)
            )
        if self._kinematic_indices is None or getattr(self, "_kinematic_articulations", None) != selected:
            self._kinematic_indices = wp.array(selected, dtype=int, device=self.model.device)
            self._kinematic_articulations = selected
            self._live_cache.pop("ik", None)
        source = shown or state
        if self._live_cache.get("ik") != id(source):
            eval_ik(
                self.model,
                source,
                self._kinematic_state.joint_q,
                self._kinematic_state.joint_qd,
                indices=self._kinematic_indices,
            )
            self._live_cache["ik"] = id(source)
        return self._kinematic_state

    def _joint_edit_mask(self, joint: int) -> tuple[wp.array[bool] | None, wp.array[int] | None]:
        """Select the edited joint and its descendants, leaving sibling branches untouched."""
        key = ("edit_branch", joint)
        if key not in self._cache:
            for previous in tuple(self._cache):
                if previous[0] == "edit_branch":
                    del self._cache[previous]
            model = self.model
            art = int(self.read(model, "joint_articulation", joint)[0])
            if art < 0:
                return None, None
            first = int(self.read(model, "articulation_start", art)[0])
            end = int(self.read(model, "articulation_end", art)[0])
            ancestors = self.read(model, "joint_ancestor", first, end - first)
            owners = self.read(model, "joint_articulation", first, end - first)
            selected = np.zeros(end - first, dtype=bool)
            for local, ancestor in enumerate(ancestors):
                selected[local] = owners[local] == art and (
                    first + local == joint or (first <= ancestor < first + local and selected[ancestor - first])
                )
            mask = wp.zeros(model.joint_count, dtype=bool, device=model.device)
            mask[first:end].assign(selected)
            indices = wp.array([art], dtype=int, device=model.device)
            self._cache[key] = mask, indices
        return self._cache[key]

    def apply_edit(
        self,
        owner: object,
        attr: str,
        start: int,
        values: ArrayLike,
        *,
        joint_index: int | None = None,
        save_undo: bool = True,
    ) -> bool:
        """Validate and apply an inspector edit without replacing device allocations.

        Control edits are allowed at the live frame; state edits also require a
        pause. Joint coordinate/velocity edits update only the joint's descendant
        branch through masked forward kinematics, then synchronize the example's
        alternate state buffer. History is refreshed after paused edits, never
        used as a starting point for resimulation.

        Args:
            owner: Live component or namespace container holding the array.
            attr: Unqualified array attribute on the owner.
            start: First storage row to replace; count is inferred from values.
            values: Finite replacement values, converted to the stored dtype.
            joint_index: Owning joint for coordinate/DOF edits. If omitted, a
                single selected joint may supply it.
            save_undo: Save the previous values. Drag gestures manage their own
                initial undo snapshot and pass False for intermediate changes.

        Returns:
            True if applied; False if editing is disallowed or validation fails.
        """
        control = self.is_control_owner(owner)
        if not (self.can_edit_control() if control else self.can_edit_state()):
            self.message = (
                "Return to the live frame to edit Control inputs." if control else "Pause at the live frame to edit."
            )
            return False
        array = getattr(owner, attr, None)
        if not isinstance(array, wp.array):
            return False
        if control:
            self._capture_control_initial()
            # Optional/custom arrays can be installed after example creation.
            # Preserve their values before the first edit as well.
            for name, value in _component_arrays(self._control_owner).items():
                if name not in self._control_initial:
                    self._control_initial[name] = wp.clone(value, requires_grad=False)
        row_size = int(np.prod((*array.shape[1:], *getattr(array.dtype, "_shape_", ())), dtype=int))
        count = max(1, np.asarray(values).size // max(1, row_size))
        if start < 0 or start + count > len(array):
            return False
        target = array[start : start + count]
        # Preserve the original scalar type, including integer and boolean custom attributes.
        previous = target.numpy().copy()
        values = np.asarray(values, dtype=previous.dtype).reshape(previous.shape)
        if not np.isfinite(values).all():
            self.message = "Enter finite values."
            return False
        indices = mask = None
        joint_state = isinstance(owner, State) and attr in ("joint_q", "joint_qd")
        if joint_state:
            supported, reason = self._joint_state_support()
            if not supported:
                self.message = reason
                return False
            if joint_index is None and self.selected and self.selected[0] == "joint":
                joint_index = self.selected[1]
            if joint_index is None:
                return False
            mask, indices = self._joint_edit_mask(joint_index)
            if mask is None or int(self.read(self.model, "joint_type", joint_index)[0]) == int(JointType.ROD):
                return False
            if self._uses_maximal_coordinates(self.bindings.solver):
                eval_ik(self.model, owner, owner.joint_q, owner.joint_qd, indices=indices)
                previous = target.numpy().copy()
        q_prev_target = prev_q_prev = None
        if joint_state and attr == "joint_q" and isinstance(getattr(owner, "joint_q_prev", None), wp.array):
            if start + count <= len(owner.joint_q_prev):
                q_prev_target = owner.joint_q_prev[start : start + count]
                prev_q_prev = q_prev_target.numpy().copy()
        target.assign(values)
        if q_prev_target is not None:
            q_prev_target.assign(values)
        if joint_state:
            eval_fk(self.model, owner.joint_q, owner.joint_qd, owner, indices=indices, joint_mask=mask)
        if isinstance(owner, State):
            notify_state = getattr(self.bindings.solver, "notify_state_changed", None)
            if callable(notify_state):
                try:
                    notify_state(owner)
                except Exception as exc:
                    target.assign(previous)
                    if q_prev_target is not None and prev_q_prev is not None:
                        q_prev_target.assign(prev_q_prev)
                    if joint_state:
                        eval_fk(self.model, owner.joint_q, owner.joint_qd, owner, indices=indices, joint_mask=mask)
                    self._live_cache.clear()
                    self.message = f"Solver rejected state edit: {exc}"
                    return False
        if save_undo:
            self._undo = (owner, attr, start, previous, joint_index)
        # Running edits are recorded with the next simulation frame. Avoid
        # downloading a second full snapshot on every control drag event.
        if self.history is not None and self.viewer.is_paused():
            self.history.replace_next_frame()
        if not control and (authoring := getattr(self, "authoring", None)):
            authoring._placement_query = None
        self._live_cache.clear()
        self._drafts.clear()
        self.message = ""
        return True
