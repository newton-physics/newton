# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect live acceleration structures with on-demand model BVH refitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..geometry.tri_mesh_collision import TriMeshCollisionDetector
from ..geometry.types import GeoType
from ..sim.collide import CollisionPipeline
from .viewer_session import ViewerSession

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any

    from newton import Model, State
    from newton._src.viewer.viewer_model_tree import ModelTree
    from newton._src.viewer.viewer_types import TreeKey, ViewerExample
    from newton._src.viewer.viewer_workspace import ViewerWorkspace
    from newton.viewer import ViewerBase


logger = logging.getLogger(__name__)


@wp.kernel
def _query_bounds(
    resource: wp.uint64, mesh: bool, group: int, query_count: int, indices: wp.array[int], count: wp.array[int]
) -> None:
    query_index = wp.tid()
    # When detail is limited, give each world its own traversal and quota so
    # early worlds cannot consume the entire budget. Missing groups stay empty.
    quota = indices.shape[0] // query_count
    if query_index < indices.shape[0] % query_count:
        quota += 1
    if query_count > 1:
        group = query_index
    bvh = resource
    if mesh:
        bvh = wp.mesh_get_bvh(resource)
    root = int(-1)
    if group >= 0:
        root = wp.bvh_get_group_root(bvh, group)
        if root < 0:
            return
    query = wp.bvh_query_aabb(bvh, wp.vec3(-1.0e30), wp.vec3(1.0e30), root)
    item = int(0)
    size = int(0)
    while size < quota and wp.bvh_query_next(query, item):
        indices[wp.atomic_add(count, 0, 1)] = item
        size += 1


@wp.kernel
def _bounds_lines(
    indices: wp.array[int],
    count: wp.array[int],
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    triangles: wp.array[int],
    groups: wp.array[int],
    mesh: bool,
    shape: int,
    shape_body: wp.array[int],
    shape_pose: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_world: wp.array[int],
    body_pose: wp.array[wp.transform],
    offsets: wp.array[wp.vec3],
    visible_worlds: wp.array[int],
    layer: wp.transform,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
) -> None:
    slot, edge = wp.tid()
    line = slot * 12 + edge
    starts[line] = wp.vec3(0.0)
    ends[line] = wp.vec3(0.0)
    if slot >= count[0]:
        return
    item = indices[slot]
    world = int(-1)
    if shape >= 0:
        world = shape_world[shape]
    elif groups.shape[0] > item:
        world = groups[item]
    if world >= 0 and world < visible_worlds.shape[0] and visible_worlds[world] == 0:
        return
    lower, upper = wp.vec3(0.0), wp.vec3(0.0)
    if mesh:
        a = vertices[triangles[item * 3]]
        b = vertices[triangles[item * 3 + 1]]
        c = vertices[triangles[item * 3 + 2]]
        lower, upper = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
    else:
        lower, upper = wp.vec3(lowers[item]), wp.vec3(uppers[item])
    # Four edges parallel to each coordinate axis.
    axis, corner = edge // 4, edge % 4
    start = wp.vec3(lower)
    end = wp.vec3(lower)
    for component in range(3):
        if component == axis:
            end[component] = upper[component]
        else:
            bit = (component - axis + 2) % 3
            if corner & (1 << bit):
                start[component] = upper[component]
                end[component] = upper[component]
    if shape >= 0:
        pose = shape_pose[shape]
        body = shape_body[shape]
        if body >= 0:
            pose = body_pose[body] * pose
        start = wp.transform_point(pose, wp.cw_mul(start, shape_scale[shape]))
        end = wp.transform_point(pose, wp.cw_mul(end, shape_scale[shape]))
    if world >= 0 and world < offsets.shape[0]:
        start += offsets[world]
        end += offsets[world]
    starts[line] = wp.transform_point(layer, start)
    ends[line] = wp.transform_point(layer, end)


@wp.kernel
def _frame_bounds(
    group: int,
    enabled_shapes: wp.array[wp.uint32],
    shape_type: wp.array[int],
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    triangles: wp.array[int],
    groups: wp.array[int],
    mesh: bool,
    shape: int,
    shape_body: wp.array[int],
    shape_pose: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_world: wp.array[int],
    body_pose: wp.array[wp.transform],
    offsets: wp.array[wp.vec3],
    visible_worlds: wp.array[int],
    layer: wp.transform,
    bounds: wp.array2d[float],
) -> None:
    item = wp.tid()
    if group >= 0 and groups[item] != group:
        return
    # Infinite planes have artificial finite BVH bounds, unsuitable for framing.
    if enabled_shapes.shape[0] > item and shape_type[enabled_shapes[item]] == int(GeoType.PLANE):
        return
    world = int(-1)
    if shape >= 0:
        world = shape_world[shape]
    elif groups.shape[0] > item:
        world = groups[item]
    if world >= 0 and world < visible_worlds.shape[0] and visible_worlds[world] == 0:
        return
    lower, upper = wp.vec3(0.0), wp.vec3(0.0)
    if mesh:
        a = vertices[triangles[item * 3]]
        b = vertices[triangles[item * 3 + 1]]
        c = vertices[triangles[item * 3 + 2]]
        lower, upper = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
    else:
        lower, upper = wp.vec3(lowers[item]), wp.vec3(uppers[item])
    if not wp.isfinite(lower) or not wp.isfinite(upper):
        return
    pose = wp.transform_identity()
    scale = wp.vec3(1.0)
    if shape >= 0:
        pose = shape_pose[shape]
        body = shape_body[shape]
        if body >= 0:
            pose = body_pose[body] * pose
        scale = shape_scale[shape]
    offset = wp.vec3(0.0)
    if world >= 0 and world < offsets.shape[0]:
        offset = offsets[world]
    for corner in range(8):
        point = wp.vec3(
            upper[0] if corner & 1 else lower[0],
            upper[1] if corner & 2 else lower[1],
            upper[2] if corner & 4 else lower[2],
        )
        point = wp.transform_point(layer, wp.transform_point(pose, wp.cw_mul(point, scale)) + offset)
        for axis in range(3):
            wp.atomic_min(bounds, 0, axis, point[axis])
            wp.atomic_max(bounds, 1, axis, point[axis])


@dataclass
class BvhResource:
    owner: object
    attribute: str
    usage: str
    local: bool = False
    index: int | None = None

    def get(self) -> wp.Bvh | wp.Mesh | None:
        value = getattr(self.owner, self.attribute, None)
        if self.index is not None:
            value = value[self.index] if value is not None and self.index < len(value) else None
        return value


class BvhResources:
    """Discover host-owned BVHs lazily; only a selected BVH adds GPU work."""

    root = ("bvh_collection", 0)
    meshes = ("bvh_collection", "meshes")
    page_size = 128

    def __init__(
        self, model: Model, example: ViewerExample | None = None, *, session: ViewerSession | None = None
    ) -> None:
        self.model, self.example = model, example
        self.session = session if session is not None else ViewerSession.from_example(example)
        self.entries = {}
        self._mesh_ids = None
        self._shape_pages = {}
        self._shape_pointers = None
        self._owners = {}
        self._instances = {}
        self.show_bounds = True
        self.limit = 256
        self.scope = 1
        self.world = 0
        self._buffers = None
        self._focus_buffers = None
        self._graph = self._signature = None
        self._graphs = {}
        self._line_name = None
        self.error = ""

    @staticmethod
    def handles(key: TreeKey) -> bool:
        return key is not None and (key[0].startswith("bvh") or key[0] == "shape_bvh")

    @staticmethod
    def _node(tree: ModelTree, key: TreeKey, label: str, parent: TreeKey | None = None, *, world: int = -1) -> bool:
        fresh = key not in tree.labels
        tree.labels[key] = label
        tree.meta[key] = {"world": world}
        if parent is not None:
            tree.parents[key] = parent
            if key not in tree.children[parent]:
                tree.children[parent].append(key)
        return fresh

    def attach(self, tree: ModelTree) -> None:
        if not (
            self.model.bvh_shapes
            or self.model.bvh_particles
            or self.model._mesh_keep_alive
            or self.model.heightfield_meshes
        ):
            return
        if self._node(tree, self.root, "BVHs"):
            # A model-owned query resource spans all worlds, including worlds
            # whose explorer branches have not been expanded yet.
            self.scope = 1 if any(key[0] == "root" and key[1] >= 0 for key in tree.roots) else 2
            tree.roots.append(self.root)
        for name, label in (("shapes", "Shape queries"), ("particles", "Particle queries")):
            attr = "bvh_" + name
            if getattr(self.model, attr) is not None:
                key = ("bvh", name)
                self.entries[key] = BvhResource(self.model, attr, "Raycast and sensor queries")
                self._node(tree, key, label, self.root)
        count = len(self.model._mesh_keep_alive) + len(self.model.heightfield_meshes)
        if count:
            self._node(tree, self.meshes, f"Mesh BVHs · {count:,}", self.root)

    def _mesh(self, index: int) -> BvhResource:
        count = len(self.model._mesh_keep_alive)
        attr, local = ("_mesh_keep_alive", index) if index < count else ("heightfield_meshes", index - count)
        return BvhResource(self.model, attr, "Triangle raycasts and collision narrowphase", local=True, index=local)

    def expand(self, tree: ModelTree, key: TreeKey) -> None:
        if key == self.root:
            self.attach(tree)
            self._discover(tree)
        elif key == self.meshes or key[0] == "bvh_page":
            if tree.children.get(key):
                return
            first, end = (
                (0, len(self.model._mesh_keep_alive) + len(self.model.heightfield_meshes))
                if key == self.meshes
                else key[1]
            )
            size = self.page_size
            while end - first > size * self.page_size:
                size *= self.page_size
            for start in range(first, end, size if end - first > self.page_size else 1):
                if end - first > self.page_size:
                    stop = min(start + size, end)
                    self._node(tree, ("bvh_page", (start, stop)), f"Meshes {start}-{stop - 1}", key)
                else:
                    node = ("bvh_mesh", start)
                    self.entries[node] = self._mesh(start)
                    self._node(tree, node, f"Mesh BVH {start}", key)

    def _discover(self, tree: ModelTree) -> None:
        # Inspect known owners, never recursively traverse a model's object graph.
        seen = set()
        for name, pipeline in self.session.collision_pipelines(self.model):
            detector = pipeline._soft_self_contact_detector
            if detector is not None and id(detector) not in seen:
                seen.add(id(detector))
                self._add_detector(tree, detector, name)
        for owner in (self.example, self.model, self.session.solver):
            for name, value in vars(owner).items() if hasattr(owner, "__dict__") else ():
                if id(value) in seen:
                    continue
                seen.add(id(value))
                if isinstance(value, CollisionPipeline) and value.model is self.model:
                    detector = value._soft_self_contact_detector
                    if detector is not None and id(detector) not in seen:
                        seen.add(id(detector))
                        self._add_detector(tree, detector, name)
                elif isinstance(value, TriMeshCollisionDetector) and value.model is self.model:
                    self._add_detector(tree, value, name)
                context = getattr(value, "_render_context", None)
                triangle_attr = "_triangle_mesh"
                # The deprecated tiled camera still owns a separate surface BVH.
                if context is None and getattr(value, "model", None) is self.model:
                    context = getattr(value, "_SensorTiledCamera__render_context", None)
                    triangle_attr = "triangle_mesh"
                if context is not None and getattr(context, "_model", self.model) is self.model:
                    if getattr(context, triangle_attr, None) is not None:
                        node = ("bvh", (name, "surface"))
                        self.entries[node] = BvhResource(context, triangle_attr, "Sensor deformable surface raycasts")
                        self._node(tree, node, name + " · Surface triangles", self.root)
        for shape, source in enumerate(self.model.shape_source):
            if getattr(source, "bvh", None) is not None and id(source) not in seen:
                seen.add(id(source))
                key = ("bvh_gaussian", shape)
                self.entries[key] = BvhResource(source, "bvh", "Gaussian raycasts", local=True)
                self._node(tree, key, f"Gaussian BVH · Shape {shape}", self.root)

    def _add_detector(self, tree: ModelTree, detector: object, name: str) -> None:
        group = ("bvh_collection", ("collision", name))
        self._node(tree, group, name.lstrip("_") + " · Self-collision", self.root)
        for attr, label in (("bvh_tris", "Triangle bounds"), ("bvh_edges", "Edge bounds")):
            if getattr(detector, attr, None) is not None:
                node = ("bvh", (name, attr))
                self.entries[node] = BvhResource(detector, attr, "Soft self-collision")
                self._node(tree, node, label, group)

    def add_shape(self, tree: ModelTree, shape: int) -> None:
        if self.model.shape_source[shape] is None:
            return
        if self._mesh_ids is None:
            self._mesh_ids = {
                int(mesh.id): i for i, mesh in enumerate((*self.model._mesh_keep_alive, *self.model.heightfield_meshes))
            }
        pointers = self.model.shape_source_ptr
        page = shape // self.page_size
        if page not in self._shape_pages:
            start = page * self.page_size
            self._shape_pages[page] = pointers[start : min(start + self.page_size, len(pointers))].numpy()
        mesh = self._mesh_ids.get(int(self._shape_pages[page][shape % self.page_size]))
        source = self.model.shape_source[shape]
        entry = self._mesh(mesh) if mesh is not None else BvhResource(source, "bvh", "Gaussian raycasts", local=True)
        if entry.get() is not None:
            key = ("shape_bvh", shape)
            self.entries[key] = entry
            self._node(
                tree,
                key,
                "Triangle BVH" if mesh is not None else "Gaussian BVH",
                ("shape", shape),
                world=tree.world_for(("shape", shape)),
            )

    def reveal(self, tree: ModelTree, key: TreeKey) -> None:
        if key[0] == "shape_bvh":
            tree.ensure_world(tree.world_for(("shape", key[1])))
            self.add_shape(tree, key[1])
        elif key[0] == "bvh_mesh":
            parent = self.meshes
            while key not in tree.labels:
                self.expand(tree, parent)
                parent = next(
                    (p for p in tree.children[parent] if p[0] == "bvh_page" and p[1][0] <= key[1] < p[1][1]), key
                )
        else:
            self.expand(tree, self.root)

    def _shape(self, ws: ViewerWorkspace, key: TreeKey, entry: BvhResource, resource: wp.Bvh | wp.Mesh) -> int:
        if not entry.local:
            return -1
        if key[0] in ("shape_bvh", "bvh_gaussian"):
            return key[1]
        if self._shape_pointers is None:
            self._shape_pointers = self.model.shape_source_ptr.numpy()
        if resource.id not in self._owners:
            self._owners[resource.id] = np.flatnonzero(self._shape_pointers == resource.id)
        owners = self._owners[resource.id]
        if not len(owners):
            return -1
        self._instances.setdefault(
            key, next((i for i, shape in enumerate(owners) if ws.tree.world_for(("shape", int(shape))) == ws.world), 0)
        )
        return int(owners[min(self._instances[key], len(owners) - 1)])

    def draw(self, ws: ViewerWorkspace) -> None:
        w, ui, key = ws.widgets, ws.gui.ui.imgui, ws.selected
        if key not in self.entries:
            if w.button("Refresh", small=True):
                self.expand(ws.tree, self.root)
                ws.tree.rebuild()
            ui.text_wrapped(
                "Existing query and collision acceleration structures. Select a BVH to inspect its primitive bounds."
            )
            return
        entry = self.entries[key]
        resource = entry.get()
        if resource is None:
            w.text_value("Status", "Unavailable")
            return
        mesh = isinstance(resource, wp.Mesh)
        count = len(resource.indices) // 3 if mesh else len(resource.lowers)
        shape = self._shape(ws, key, entry, resource)
        with w.group("BVH") as opened:
            if opened:
                w.text_value("Use", entry.usage)
                w.text_value("Source", entry.attribute)
                w.text_value("Device", str(resource.device))
                w.text_value("Coordinates", "Shape local" if entry.local else "World")
                w.number("Triangles" if mesh else "Primitives", count, integer=True)
                if not mesh:
                    w.text_value("Constructor", resource._constructor.name)
                w.boolean("World groups", resource.groups is not None)
        if entry.local and shape >= 0:
            owners = self._owners.get(resource.id, ())
            if key[0] == "bvh_mesh" and len(owners) > 1:
                self._instances[key] = int(
                    w.number(
                        "Instance",
                        self._instances[key],
                        integer=True,
                        editable=True,
                        speed=0.05,
                        bounds=(0, len(owners) - 1),
                    ).value
                )
                shape = self._shape(ws, key, entry, resource)
            label = self.model.shape_label[shape] or f"Shape {shape}"
            if w.reference("Shape", shape, target=label, caption=ws.tree.reference_caption(("shape", shape), label)):
                ws.navigate(("shape", shape))
        with w.group("Visualization") as opened:
            if opened:
                self.show_bounds = w.boolean("Show bounds", self.show_bounds, editable=True).value
                self.limit = int(
                    w.number("Bounds budget", self.limit, integer=True, editable=True, speed=1, bounds=(16, 2048)).value
                )
                if not entry.local and resource.groups is not None:
                    self.scope = w.choice(
                        "Scope", self.scope, {0: "World", 1: "All worlds", 2: "Shared"}, editable=True
                    ).value
                    if self.scope == 0:
                        self.world = int(
                            w.number(
                                "World",
                                self.world,
                                integer=True,
                                editable=True,
                                speed=0.05,
                                bounds=(0, max(0, self.model.world_count - 1)),
                            ).value
                        )
                ui.text_wrapped("Primitive bounds, not internal tree nodes. Increase the budget to show more detail.")
                if not entry.local and resource.groups is not None and self.scope == 1:
                    ui.text_wrapped("Distributes detail across worlds, with at least one bound per nonempty world.")
                if resource is self.model.bvh_shapes or resource is self.model.bvh_particles:
                    ui.text_wrapped(
                        "Refits to the displayed state while selected; preserves the existing tree topology."
                    )
                else:
                    ui.text_wrapped("Reads the live acceleration structure maintained by its owner.")
                if self.error:
                    ui.text_wrapped(self.error)
                if not ws._at_live_frame():
                    ui.text_wrapped("BVHs are not recorded in history. Return to the live frame to visualize them.")
                elif entry.local and shape < 0:
                    ui.text_wrapped("No shape instance is available for this local-space BVH.")

    def overlay(self, ws: ViewerWorkspace) -> None:
        try:
            self._update_overlay(ws)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.show_bounds = False
            self.close(ws.viewer)
            logger.exception("Failed to visualize BVH bounds")

    def _group(self, entry: BvhResource, resource: wp.Bvh | wp.Mesh) -> int:
        if entry.local or resource.groups is None or self.scope == 1:
            return -1
        return self.model.world_count if self.scope == 2 else self.world

    @staticmethod
    def _state(ws: ViewerWorkspace) -> State | None:
        return (
            getattr(ws.viewer, "_last_state", None)
            or getattr(ws.example, "state_0", None)
            or getattr(ws.example, "state", None)
        )

    def _refit(self, resource: wp.Bvh | wp.Mesh, state: State | None) -> None:
        # Most examples never query these model BVHs after finalization. Keep
        # the selected one current, but do not alter solver-owned structures.
        if state is not None:
            if resource is self.model.bvh_shapes:
                self.model.bvh_refit_shapes(state)
            elif resource is self.model.bvh_particles:
                self.model.bvh_refit_particles(state)

    def _refit_signature(self, resource: wp.Bvh | wp.Mesh, state: State | None) -> tuple[Hashable, ...]:
        if state is None:
            return ()
        model = self.model
        if resource is model.bvh_shapes:
            values = (
                model.bvh_shape_count_enabled,
                model.bvh_shape_enabled,
                model.bvh_shape_bounds,
                model.bvh_shape_world_transforms,
                model.shape_type,
            )
        elif resource is model.bvh_particles:
            # Particle state may alternate independently of body state.
            values = (state.particle_q, model.particle_radius, model.particle_world)
        else:
            return ()
        return (model.world_count, *(value.ptr if isinstance(value, wp.array) else value for value in values))

    def _geometry_args(
        self,
        ws: ViewerWorkspace,
        resource: wp.Bvh | wp.Mesh,
        shape: int,
        empty_vec: wp.array[wp.vec3],
        empty_int: wp.array[int],
    ) -> list[Any]:
        viewer, model = ws.viewer, self.model
        mesh = isinstance(resource, wp.Mesh)
        state = self._state(ws)
        visible_worlds = getattr(viewer, "_visible_worlds_mask", None)
        return [
            empty_vec if mesh else resource.lowers,
            empty_vec if mesh else resource.uppers,
            resource.points if mesh else empty_vec,
            resource.indices if mesh else empty_int,
            resource.groups if resource.groups is not None else empty_int,
            mesh,
            shape,
            model.shape_body,
            model.shape_transform,
            model.shape_scale,
            model.shape_world,
            state.body_q if state is not None else model.body_q,
            viewer.world_offsets if viewer.world_offsets is not None else empty_vec,
            visible_worlds if visible_worlds is not None else empty_int,
            viewer.layer.xform,
        ]

    def bounds(self, ws: ViewerWorkspace) -> np.ndarray | None:
        """Reduce the full selected BVH on demand, reading back only six floats."""
        entry = self.entries.get(ws.selected)
        if entry is None or not ws._at_live_frame():
            return None
        resource = entry.get()
        if resource is None or resource.device != self.model.device:
            return None
        shape = self._shape(ws, ws.selected, entry, resource)
        if entry.local and shape < 0:
            return None
        count = len(resource.indices) // 3 if isinstance(resource, wp.Mesh) else len(resource.lowers)
        if not count:
            return None
        self._refit(resource, self._state(ws))
        device = resource.device
        if self._focus_buffers is None or self._focus_buffers[0].device != device:
            self._focus_buffers = (
                wp.empty((2, 3), dtype=float, device=device),
                wp.empty(0, dtype=wp.vec3, device=device),
                wp.empty(0, dtype=int, device=device),
                wp.empty(0, dtype=wp.uint32, device=device),
            )
        bounds, empty_vec, empty_int, empty_uint = self._focus_buffers
        bounds[0:1].fill_(1e30)
        bounds[1:2].fill_(-1e30)
        wp.launch(
            _frame_bounds,
            dim=count,
            inputs=[
                self._group(entry, resource),
                self.model.bvh_shape_enabled if resource is self.model.bvh_shapes else empty_uint,
                self.model.shape_type,
                *self._geometry_args(ws, resource, shape, empty_vec, empty_int),
            ],
            outputs=[bounds],
            device=device,
        )
        result = bounds.numpy()
        return result if np.isfinite(result).all() and (result[1] >= result[0]).all() else None

    def _update_overlay(self, ws: ViewerWorkspace) -> None:
        # A single batch is hidden immediately on deselection or history playback.
        if self._line_name in ws.viewer.lines:
            ws.viewer.lines[self._line_name].hidden = True
        key = ws.selected
        if not self.show_bounds or key not in self.entries or not ws._at_live_frame():
            return
        entry = self.entries[key]
        resource = entry.get()
        if resource is None:
            return
        shape = self._shape(ws, key, entry, resource)
        if entry.local and shape < 0:
            return
        mesh = isinstance(resource, wp.Mesh)
        count = len(resource.indices) // 3 if mesh else len(resource.lowers)
        budget = min(count, max(16, min(2048, self.limit)))
        if not budget:
            return
        viewer, device = ws.viewer, resource.device
        if device != viewer.device:
            return
        group = self._group(entry, resource)
        query_count = 1
        if group < 0 and not entry.local and resource.groups is not None and budget < count:
            world_groups = self.model.world_count + 1  # Includes Shared.
            budget = min(count, max(budget, world_groups))
            if budget < count:
                query_count = world_groups
        if self._buffers is None or len(self._buffers[0]) != budget or self._buffers[0].device != device:
            self._buffers = (
                wp.empty(budget, dtype=int, device=device),
                wp.zeros(1, dtype=int, device=device),
                wp.empty(budget * 12, dtype=wp.vec3, device=device),
                wp.empty(budget * 12, dtype=wp.vec3, device=device),
                wp.full(budget * 12, wp.vec3(1.0, 0.43, 0.07), dtype=wp.vec3, device=device),
                wp.empty(0, dtype=wp.vec3, device=device),
                wp.empty(0, dtype=int, device=device),
            )
            self._signature = None
            self._graphs.clear()
        indices, sizes, starts, ends, colors, empty_vec, empty_int = self._buffers
        state = self._state(ws)
        args = [indices, sizes, *self._geometry_args(ws, resource, shape, empty_vec, empty_int)]
        signature = (
            resource.id,
            group,
            query_count,
            # Includes the visibility-mask allocation, which is replaced when
            # worlds are toggled. Never replay a graph bound to the old mask.
            tuple(value.ptr if isinstance(value, wp.array) else value for value in args[:-1]),
            tuple(viewer.layer.xform),
            self._refit_signature(resource, state),
        )

        def launch() -> None:
            self._refit(resource, state)
            sizes.zero_()
            wp.launch(
                _query_bounds,
                dim=query_count,
                inputs=[resource.id, mesh, group, query_count, indices, sizes],
                device=device,
            )
            wp.launch(_bounds_lines, dim=(budget, 12), inputs=args, outputs=[starts, ends], device=device)

        if device.is_cuda:
            if signature != self._signature:
                if signature not in self._graphs:
                    # Retain both state buffers used by double-buffered solvers.
                    # Compile before capture; never allocate in the replay path.
                    launch()
                    with wp.ScopedCapture(device=device) as capture:
                        launch()
                    if len(self._graphs) >= 2:
                        self._graphs.pop(next(iter(self._graphs)))
                    self._graphs[signature] = capture.graph
                self._graph, self._signature = self._graphs[signature], signature
            wp.capture_launch(self._graph)
        else:
            launch()
        self._line_name = viewer._qualify("bvh_bounds")
        if self._line_name not in viewer.lines or viewer.lines[self._line_name].max_lines < len(starts):
            from .gl.opengl import LinesGL  # noqa: PLC0415

            previous = viewer.lines.pop(self._line_name, None)
            if previous is not None:
                previous.destroy()
            # Keep diagnostic geometry on the device even when ordinary user
            # line batches use host uploads. LinesGL has a non-interop fallback.
            viewer.lines[self._line_name] = LinesGL(len(starts), device, enable_cuda_interop=True)
        viewer.log_lines(self._line_name, starts, ends, colors)
        self.error = ""

    def close(self, viewer: ViewerBase) -> None:
        previous = viewer.lines.pop(self._line_name, None)
        if previous is not None:
            previous.destroy()
        self._line_name = None
        self._graphs.clear()
        self._focus_buffers = None
        self._graph = self._buffers = self._signature = None
