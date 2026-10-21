# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lazy deformable topology and selected geometry for the local inspector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..geometry.flags import ParticleFlags
from ..sim.model import Model
from ..sim.state import State

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import ArrayLike

    from newton._src.viewer.viewer_model_tree import ModelTree
    from newton._src.viewer.viewer_types import TreeKey, ViewerExample
    from newton._src.viewer.viewer_workspace import ViewerWorkspace
    from newton.viewer import ViewerBase


@dataclass
class DeformableObject:
    """Explorer grouping over catalog particle IDs and primary-Model element indices.

    A grouping does not own simulation arrays. Ranges stay lazy until inspection
    or selection needs explicit indices, including loose-particle world groups.
    """

    kind: str
    label: str
    world: int
    particles: range | np.ndarray
    triangles: range | np.ndarray
    tetrahedra: range | np.ndarray


class ParticleSource:
    """Map a particle Model into a disjoint catalog ID range.

    Coupled examples can log particles from an auxiliary Model. ``offset`` is
    added only to explorer identities; simulation arrays keep their Model-local
    indices. Logged points identify the displayed State without copying it.
    """

    def __init__(self, model: Model, offset: int) -> None:
        self.model = model
        self.offset = offset
        self.end = offset + model.particle_count
        self.starts = model.particle_world_start.numpy()
        self.points = self.state = None
        self.hidden = True
        self.visibility_key = None

    def ranges(self, world: int) -> list[tuple[int, int]]:
        starts = self.starts
        pairs = (
            ((0, starts[0]), (starts[-2], starts[-1]))
            if world < 0
            else ((starts[world], starts[world + 1]),)
            if world < self.model.world_count
            else ()
        )
        return [(int(a), int(b)) for a, b in pairs if b > a]

    def world_for(self, index: int) -> int:
        world = int(np.searchsorted(self.starts, index, side="right")) - 1
        return world if 0 <= world < self.model.world_count else -1


@wp.kernel
def gather_display_particles(
    indices: wp.array[int],
    positions: wp.array[wp.vec3],
    worlds: wp.array[int],
    offsets: wp.array[wp.vec3],
    layer: wp.transform,
    output: wp.array[wp.vec3],
) -> None:
    i = wp.tid()
    particle = indices[i]
    point = positions[particle]
    world = worlds[particle]
    if world >= 0 and world < offsets.shape[0]:
        point += offsets[world]
    output[i] = wp.transform_point(layer, point)


@wp.kernel
def particle_bounds(points: wp.array[wp.vec3], bounds: wp.array2d[float]) -> None:
    point = points[wp.tid()]
    for axis in range(3):
        wp.atomic_min(bounds, 0, axis, point[axis])
        wp.atomic_max(bounds, 1, axis, point[axis])


@wp.kernel
def _visible_particle_bounds(points: wp.array[wp.vec3], radii: wp.array[float], bounds: wp.array2d[float]) -> None:
    i = wp.tid()
    if radii[i] > 0.0:
        for axis in range(3):
            wp.atomic_min(bounds, 0, axis, points[i][axis] - radii[i])
            wp.atomic_max(bounds, 1, axis, points[i][axis] + radii[i])


@wp.kernel
def _gather_markers(indices: wp.array[int], points: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
    i = wp.tid()
    output[i] = points[indices[i]]


@wp.kernel
def _display_all_particles(
    positions: wp.array[wp.vec3],
    worlds: wp.array[int],
    offsets: wp.array[wp.vec3],
    layer: wp.transform,
    radii: wp.array[float],
    flags: wp.array[int],
    visible: wp.array[int],
    points: wp.array[wp.vec3],
    display_radii: wp.array[float],
) -> None:
    i = wp.tid()
    point = positions[i]
    world = worlds[i]
    if world >= 0 and world < offsets.shape[0]:
        point += offsets[world]
    points[i] = wp.transform_point(layer, point)
    active = flags.shape[0] == 0 or (flags[i] & int(ParticleFlags.ACTIVE)) != 0
    display_radii[i] = radii[i] if visible[i] != 0 and active else 0.0


@wp.kernel
def _particle_mask(radii: wp.array[float], mask: wp.array[int]) -> None:
    i = wp.tid()
    mask[i] = int(radii[i] > 0.0)


@wp.kernel
def _compact_display_particles(
    points: wp.array[wp.vec3],
    radii: wp.array[float],
    mask: wp.array[int],
    offsets: wp.array[int],
    compact_points: wp.array[wp.vec3],
    compact_radii: wp.array[float],
) -> None:
    i = wp.tid()
    if mask[i] != 0:
        slot = offsets[i] - 1
        compact_points[slot] = points[i]
        compact_radii[slot] = radii[i]


@wp.func
def _display_position(
    index: int, positions: wp.array[wp.vec3], worlds: wp.array[int], offsets: wp.array[wp.vec3]
) -> wp.vec3:
    point = positions[index]
    world = worlds[index]
    if world >= 0 and world < offsets.shape[0]:
        point += offsets[world]
    return point


@wp.kernel
def _pick_deformable(
    positions: wp.array[wp.vec3],
    worlds: wp.array[int],
    offsets: wp.array[wp.vec3],
    world_mask: wp.array[int],
    display_mask: wp.array[int],
    flags: wp.array[int],
    radii: wp.array[float],
    triangles: wp.array2d[int],
    opacity: wp.array[float],
    particles: bool,
    origin: wp.vec3,
    direction: wp.vec3,
    pixel_radius: float,
    distances: wp.array[float],
    nearest: wp.array[float],
) -> None:
    i = wp.tid()
    particle = i
    if not particles:
        particle = triangles[i, 0]
    world = worlds[particle]
    distance = float(1.0e10)
    visible = world < 0 or world_mask.shape[0] == 0 or world_mask[world] != 0
    visible = visible and (display_mask.shape[0] == 0 or display_mask[particle] != 0)
    active = flags.shape[0] == 0 or (flags[particle] & int(ParticleFlags.ACTIVE)) != 0
    if visible and (not particles or active):
        a = _display_position(particle, positions, worlds, offsets)
        if particles:
            along = wp.dot(a - origin, direction)
            radius = wp.max(radii[i], along * pixel_radius)
            off = wp.length_sq(a - origin - along * direction)
            if along > 0.0 and off <= radius * radius:
                distance = wp.max(0.0, along - wp.sqrt(radius * radius - off))
        elif opacity[i] > 0.0:
            b = _display_position(triangles[i, 1], positions, worlds, offsets)
            c = _display_position(triangles[i, 2], positions, worlds, offsets)
            edge1, edge2 = b - a, c - a
            p = wp.cross(direction, edge2)
            det = wp.dot(edge1, p)
            if wp.abs(det) > 1.0e-12:
                inv = 1.0 / det
                t = origin - a
                u = wp.dot(t, p) * inv
                q = wp.cross(t, edge1)
                v = wp.dot(direction, q) * inv
                hit = wp.dot(edge2, q) * inv
                if u >= 0.0 and v >= 0.0 and u + v <= 1.0 and hit > 0.0:
                    distance = hit
    distances[i] = distance
    wp.atomic_min(nearest, 0, distance)


@wp.kernel
def _pick_index(distances: wp.array[float], nearest: wp.array[float], index: wp.array[int]) -> None:
    i = wp.tid()
    if distances[i] == nearest[0] and distances[i] < 1.0e10:
        wp.atomic_min(index, 0, i)


class DeformableCatalog:
    """Index deformable topology lazily and keep paged explorer identities stable.

    The primary Model supplies surfaces and volumes. Auxiliary particle-only
    Models exposed by conventional examples get disjoint particle ID ranges;
    triangle and tetrahedron indices always refer to the primary Model.
    Topology caches live for one Model layout and are recreated on scene rebuild.
    """

    page_size = 256

    def __init__(self, model: Model, example: ViewerExample | None = None) -> None:
        self.model = model
        self.example = example
        self.sources = [ParticleSource(model, 0)]
        self.particle_count = model.particle_count
        # Some coupled examples log a separate particle model explicitly.
        # Match its radius array at log_points(), never arbitrary debug points
        # or solver coloring groups. No particle state is copied here.
        seen = {id(model)}
        for value in vars(example).values() if example is not None else ():
            if (
                isinstance(value, Model)
                and id(value) not in seen
                and value.particle_count
                and not value.tri_count
                and not value.tet_count
            ):
                self.sources.append(ParticleSource(value, self.particle_count))
                self.particle_count += value.particle_count
                seen.add(id(value))
        self.objects = []
        self.groups = {}
        self.hidden = set()
        self._topology = {}
        self._loaded = set()
        self._materials = {}
        self._element_order = {}

    def source(self, particle: int) -> ParticleSource:
        return next(source for source in self.sources if source.offset <= particle < source.end)

    def world_for(self, particle: int) -> int:
        source = self.source(particle)
        return source.world_for(particle - source.offset)

    def external_bounds(self) -> np.ndarray:
        bounds = wp.array([[1e30] * 3, [-1e30] * 3], dtype=float, device=self.model.device)
        for source in self.sources[1:]:
            if source.points is not None and not source.hidden:
                wp.launch(
                    _visible_particle_bounds,
                    dim=source.model.particle_count,
                    inputs=[source.display_points, source.display_radii],
                    outputs=[bounds],
                    device=self.model.device,
                )
        return bounds.numpy()

    def logged_points(
        self, workspace: ViewerWorkspace, points: wp.array[wp.vec3] | None, radii: wp.array[float] | None, hidden: bool
    ) -> tuple[wp.array[wp.vec3] | None, wp.array[float] | None, bool]:
        source = next((source for source in self.sources[1:] if radii is source.model.particle_radius), None)
        if source is None or len(points) != source.model.particle_count:
            return points, radii, hidden
        viewer, model = workspace.viewer, source.model
        if source.points is not points:
            source.points = points
            source.state = next(
                (
                    state
                    for state in vars(self.example).values()
                    if isinstance(state, State) and state.particle_q is points
                ),
                None,
            )
        key = (
            tuple(sorted(self.hidden)),
            tuple(viewer._visible_worlds) if viewer._visible_worlds is not None else None,
        )
        if source.visibility_key != key:
            visible = np.ones(model.particle_count, dtype=np.int32)
            for obj_key in self.hidden:
                indices = np.asarray(self.objects[obj_key[1]].particles, dtype=int)
                indices = indices[(indices >= source.offset) & (indices < source.end)] - source.offset
                visible[indices] = 0
            if viewer._visible_worlds is not None:
                for world in range(model.world_count):
                    if world not in viewer._visible_worlds:
                        for a, b in source.ranges(world):
                            visible[a:b] = 0
            source.visible = wp.array(visible, dtype=int, device=model.device)
            source.any_visible = bool(visible.any())
            source.visibility_key = key
        source.hidden = hidden or not source.any_visible
        if not hasattr(source, "display_points"):
            source.display_points = wp.empty_like(points)
            source.display_radii = wp.empty_like(radii)
            source.empty_offsets = wp.empty(0, dtype=wp.vec3, device=model.device)
            source.empty_flags = wp.empty(0, dtype=int, device=model.device)
        if not source.hidden:
            wp.launch(
                _display_all_particles,
                dim=len(points),
                inputs=[
                    points,
                    model.particle_world,
                    viewer.world_offsets if viewer.world_offsets is not None else source.empty_offsets,
                    viewer.layer.xform,
                    radii,
                    model.particle_flags if model.particle_flags is not None else source.empty_flags,
                    source.visible,
                ],
                outputs=[source.display_points, source.display_radii],
                device=model.device,
            )
        return source.display_points, source.display_radii, source.hidden

    def topology(self, kind: str) -> np.ndarray:
        if kind not in self._topology:
            adjacency = self.model.soft_mesh_adjacency
            name = "indices" if kind == "tri" else "tet_indices"
            host = getattr(adjacency, name, None)
            if host is None:
                array = getattr(self.model, kind + "_indices", None)
                host = array.numpy() if array is not None else []
            self._topology[kind] = np.asarray(host, dtype=np.int32).reshape(-1, 3 if kind == "tri" else 4)
        return self._topology[kind]

    def elements_in_range(self, kind: str, start: int, end: int) -> np.ndarray:
        # One index for the scene avoids scanning every world's elements each
        # time another world or volume is expanded.
        if kind not in self._element_order:
            first = self.topology(kind)[:, 0]
            order = np.argsort(first, kind="stable")
            self._element_order[kind] = (first[order], order)
        first, order = self._element_order[kind]
        a, b = np.searchsorted(first, (start, end))
        return np.sort(order[a:b])

    def add_world(self, tree: ModelTree, world: int) -> None:
        if world in self._loaded:
            return
        self._loaded.add(world)
        particle_ranges = self.sources[0].ranges(world)
        if not any(source.ranges(world) for source in self.sources):
            return
        root = ("root", world)
        folder = ("group", world * 4 + 3)
        tree.labels[folder] = "Deformables"
        tree.parents[folder] = root
        tree.expanded.add(folder)
        triangles = self.topology("tri")
        tetrahedra = self.topology("tet")
        covered = []
        records = getattr(self.model, "_deformable_objects", ())
        for kind, label, w, p0, p1, e0, e1 in records:
            if w != world:
                continue
            tri = range(e0, e1) if kind == "surface" else self.elements_in_range("tri", p0, p1)
            tet = range(e0, e1) if kind == "volume" else range(0)
            self._add(tree, folder, DeformableObject(kind, label, world, range(p0, p1), tri, tet))
            covered.append((p0, p1))
        # Low-level add_particle/add_triangle/add_tetrahedron users also get
        # inspectable objects without requiring builder-specific labels.
        loose = []
        for a, b in particle_ranges:
            particles = np.arange(a, b, dtype=np.int32)
            keep = np.ones(b - a, dtype=bool)
            for p0, p1 in covered:
                keep[max(a, p0) - a : max(a, min(b, p1)) - a] = False
            particles = particles[keep]
            if not len(particles):
                continue
            tri = self.elements_in_range("tri", a, b)
            tri = tri[keep[triangles[tri, 0] - a]]
            tet = self.elements_in_range("tet", a, b)
            tet = tet[keep[tetrahedra[tet, 0] - a]]
            mesh_particles = np.unique(np.concatenate((triangles[tri].ravel(), tetrahedra[tet].ravel())))
            if len(mesh_particles):
                kind = "volume" if len(tet) else "surface"
                label = "Tet mesh" if len(tet) else "Triangle mesh"
                self._add(tree, folder, DeformableObject(kind, label, world, mesh_particles, tri, tet))
            loose.append(np.setdiff1d(particles, mesh_particles, assume_unique=True))
        for source in self.sources[1:]:
            loose.extend(np.arange(a, b, dtype=np.int32) + source.offset for a, b in source.ranges(world))
        if tree.children[folder]:
            tree.children[root].append(folder)
        if loose:
            particles = np.concatenate(loose)
            if len(particles):
                self._add(tree, root, DeformableObject("particle_set", "Particles", world, particles, (), ()))

    def _add(self, tree: ModelTree, folder: TreeKey, obj: DeformableObject) -> None:
        key = (obj.kind, len(self.objects))
        self.objects.append(obj)
        tree.labels[key] = obj.label
        tree.meta[key] = {"world": obj.world, "object": obj}
        tree.children[folder].append(key)
        tree.parents[key] = folder
        for kind, label, indices in (
            ("particle", "Particles", obj.particles),
            ("tri", "Triangles", obj.triangles),
            ("tet", "Tetrahedra", obj.tetrahedra),
        ):
            if len(indices):
                group = ("elements", len(self.groups))
                self.groups[group] = (kind, indices, key)
                tree.labels[group] = f"{label} · {len(indices):,}"
                tree.meta[group] = {"world": obj.world}
                tree.parents[group] = key
                tree.children[key].append(group)

    def expand(self, tree: ModelTree, key: TreeKey) -> None:
        if key not in self.groups or tree.children.get(key):
            return
        kind, indices, owner = self.groups[key]
        if len(indices) > self.page_size:
            for start in range(0, len(indices), self.page_size):
                end = min(len(indices), start + self.page_size)
                page = ("elements", len(self.groups))
                self.groups[page] = (kind, indices[start:end], owner)
                tree.labels[page] = f"{start + 1:,}-{end:,}"
                tree.meta[page] = {"world": self.objects[owner[1]].world}
                tree.children[key].append(page)
                tree.parents[page] = key
        else:
            for index in indices:
                child = (kind, int(index))
                title = {"particle": "Particle", "tri": "Triangle", "tet": "Tetrahedron"}[kind]
                tree.labels[child] = f"{title} {index}"
                tree.meta[child] = {"world": self.objects[owner[1]].world, "owner": owner}
                tree.children[key].append(child)
                tree.parents[child] = key

    def owner(self, tree: ModelTree, particle: int) -> TreeKey | None:
        world = tree.world_for(("particle", particle))
        tree.ensure_world(world)
        for i, obj in enumerate(self.objects):
            if obj.world == world and particle in obj.particles:
                return obj.kind, i
        return None

    def reveal(self, tree: ModelTree, key: TreeKey) -> None:
        kind, index = key
        particle = index if kind == "particle" else int(self.topology(kind)[index, 0])
        owner = self.owner(tree, particle)
        if owner is None:
            return
        for group in tree.children[owner]:
            element, indices, _ = self.groups[group]
            if element != kind or index not in indices:
                continue
            self.expand(tree, group)
            for child in tree.children[group]:
                if child in self.groups and index in self.groups[child][1]:
                    self.expand(tree, child)
                    break
            return

    def pick(
        self,
        workspace: ViewerWorkspace,
        origin: ArrayLike,
        direction: ArrayLike,
        *,
        particles: bool,
        limit: float,
        triangle: bool = False,
    ) -> TreeKey | None:
        viewer = workspace.viewer
        result = None
        # All sources compete with the rigid hit and each other in depth.
        for source in self.sources:
            primary = source is self.sources[0]
            positions = viewer._last_state.particle_q if primary else source.points
            if positions is None or (not primary and source.hidden):
                continue
            use_particles = particles if primary else True
            if not primary and not viewer.show_particles:
                continue
            display = viewer._workspace_deformables if primary else source
            hit, distance = self._pick_source(
                workspace,
                source.model,
                positions,
                display,
                origin,
                direction,
                particles=use_particles,
                limit=limit,
            )
            if hit is None:
                continue
            limit = distance
            if use_particles:
                key = ("particle", hit + source.offset)
                result = (
                    key if primary or workspace.selection_mode == "particle" else self.owner(workspace.tree, key[1])
                )
            else:
                result = ("tri", hit) if triangle else self.owner(workspace.tree, int(self.topology("tri")[hit, 0]))
        return result

    def _pick_source(
        self,
        workspace: ViewerWorkspace,
        model: Model,
        positions: wp.array[wp.vec3],
        display: DeformableDisplay | ParticleSource,
        origin: ArrayLike,
        direction: ArrayLike,
        *,
        particles: bool,
        limit: float,
    ) -> tuple[int | None, float]:
        viewer = workspace.viewer
        count = model.particle_count if particles else model.tri_count
        if not count or (not particles and not viewer.show_triangles):
            return None, limit
        device = model.device
        distances = wp.empty(count, dtype=float, device=device)
        nearest = wp.array([limit], dtype=float, device=device)
        index = wp.array([count], dtype=int, device=device)
        wp.launch(
            _pick_deformable,
            dim=count,
            inputs=[
                positions,
                model.particle_world,
                viewer.world_offsets if viewer.world_offsets is not None else wp.empty(0, dtype=wp.vec3, device=device),
                viewer._visible_worlds_mask
                if viewer._visible_worlds_mask is not None
                else wp.empty(0, dtype=int, device=device),
                display.visible if display is not None else wp.empty(0, dtype=int, device=device),
                model.particle_flags if model.particle_flags is not None else wp.empty(0, dtype=int, device=device),
                model.particle_radius,
                model.tri_indices if model.tri_indices is not None else wp.empty((0, 3), dtype=int, device=device),
                model.tri_opacity if model.tri_opacity is not None else wp.empty(0, dtype=float, device=device),
                particles,
                origin,
                direction,
                8 * np.tan(np.radians(viewer.camera.fov) / 2) / max(1, workspace.rect[3]),
                distances,
                nearest,
            ],
            device=device,
        )
        wp.launch(_pick_index, dim=count, inputs=[distances, nearest, index], device=device)
        hit = int(index.numpy()[0])
        if hit == count:
            return None, limit
        return hit, float(nearest.numpy()[0])

    def elements(
        self, key: TreeKey
    ) -> tuple[Sequence[int] | np.ndarray, Sequence[int] | np.ndarray, Sequence[int] | np.ndarray]:
        kind, index = key
        if kind in ("surface", "volume", "particle_set"):
            obj = self.objects[index]
            return obj.particles, obj.triangles, obj.tetrahedra
        if key in self.groups:
            kind, indices, _ = self.groups[key]
        else:
            indices = [index]
        if kind == "particle":
            return indices, (), ()
        if kind in ("tri", "tet"):
            return (
                np.unique(self.topology(kind)[indices]),
                indices if kind == "tri" else (),
                indices if kind == "tet" else (),
            )
        return (), (), ()

    def visible(self, tree: ModelTree, key: TreeKey) -> bool:
        if key[0] in ("surface", "volume", "particle_set"):
            return key not in self.hidden
        owner = tree.meta.get(key, {}).get("owner")
        if key in self.groups:
            owner = self.groups[key][2]
        return owner not in self.hidden

    def selection(self, keys: Iterable[TreeKey]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        particles, triangles, tetrahedra, markers = [], [], [], []
        for key in keys:
            p, tri, tet = self.elements(key)
            particles.extend(p)
            triangles.extend(tri)
            tetrahedra.extend(tet)
            if (
                key[0] == "particle"
                or (key in self.groups and self.groups[key][0] == "particle")
                or key[0] == "particle_set"
            ):
                markers.extend(p)
        return tuple(np.unique(v).astype(np.int32) for v in (particles, triangles, tetrahedra, markers))

    def properties(self, workspace: ViewerWorkspace, key: TreeKey) -> None:
        w, ui, model = workspace.widgets, workspace.gui.ui.imgui, self.model
        kind, index = key
        particles, triangles, tets = self.elements(key)
        if kind == "particle":
            source = self.source(index)
            model = source.model
            index -= source.offset
            with w.group("Particle") as opened:
                if opened:
                    w.number("Index", index, integer=True)
                    w.number("Mass", workspace.read(model, "particle_mass", index)[0], unit="kg")
                    w.number("Radius", workspace.read(model, "particle_radius", index)[0], unit="m")
                    values = workspace.read(model, "particle_flags", index)
                    flags = int(values[0]) if values.size else int(ParticleFlags.ACTIVE)
                    ui.begin_disabled()
                    ui.checkbox("Active", bool(flags & int(ParticleFlags.ACTIVE)))
                    ui.same_line()
                    ui.checkbox("Proxy", bool(flags & int(ParticleFlags.PROXY)))
                    ui.end_disabled()
            state = workspace.viewer._last_state if source is self.sources[0] else source.state
            if state is not None:
                with w.group("State") as opened:
                    if opened:
                        for label, name, unit in (
                            ("Position", "particle_q", "m"),
                            ("Velocity", "particle_qd", "m/s"),
                            ("Force", "particle_f", "N"),
                        ):
                            values = workspace.read(state, name, index, live=True)
                            if values.size:
                                w.vec3(label, values, unit=unit)
        else:
            with w.group("Topology") as opened:
                if opened:
                    w.number("Particles", len(particles), integer=True)
                    w.number("Triangles", len(triangles), integer=True)
                    w.number("Tetrahedra", len(tets), integer=True)
            if kind in ("tri", "tet"):
                with w.group("Vertices") as opened:
                    if opened:
                        for p in self.topology(kind)[index]:
                            if ui.button(f"Particle {p}"):
                                workspace.select(("particle", int(p)))
            for element, indices, title, labels in (
                ("tri", triangles, "Surface material", ("Shear", "Area stiffness", "Damping", "Drag", "Lift")),
                ("tet", tets, "Volume material", ("Shear modulus", "Lamé λ", "Damping")),
            ):
                if not len(indices):
                    continue
                with w.group(title) as opened:
                    if opened:
                        # Cache summaries: inspecting a large mesh must not
                        # reduce its complete material array every frame.
                        material_key = (key, element)
                        if material_key not in self._materials:
                            first, last = int(min(indices)), int(max(indices))
                            data = workspace.read(model, element + "_materials", first, last - first + 1)
                            values = data[np.asarray(indices, dtype=int) - first]
                            self._materials[material_key] = (values.min(axis=0), values.max(axis=0))
                        lower, upper = self._materials[material_key]
                        for column, label in enumerate(labels):
                            if column >= len(lower):
                                break
                            lo, hi = lower[column], upper[column]
                            if lo == hi:
                                w.number(label, float(lo))
                            else:
                                w.array(label, [lo, hi], labels=("Min", "Max"))
                        if kind == "tri":
                            w.number("Rest area", workspace.read(model, "tri_areas", index)[0], unit="m²")
                        if kind == "tet":
                            w.matrix3("Rest pose", workspace.read(model, "tet_poses", index))


class DeformableSelection:
    """Gather selected vertices into a compact mesh for depth-tested highlighting.

    Selection changes rebuild compact topology; frame updates gather positions
    on the device using source-local particle indices. Output vertices already
    include world spacing and the layer transform, so the outline pass uses an
    identity instance transform. Call close before discarding the selection.
    """

    def __init__(self, workspace: ViewerWorkspace, keys: Sequence[TreeKey]) -> None:
        from .gl.opengl import MeshGL  # noqa: PLC0415

        self.viewer, self.catalog = workspace.viewer, workspace.tree.deformables
        self.particles, tri, _tet, self.markers = self.catalog.selection(keys)
        faces = self.catalog.topology("tri")[tri]
        explicit_tets = []
        for key in keys:
            _, surface, volume = self.catalog.elements(key)
            if len(volume) and (key[0] != "volume" or not len(surface)):
                explicit_tets.extend(volume)
        if explicit_tets:
            cells = self.catalog.topology("tet")[np.unique(explicit_tets)]
            faces = np.concatenate((faces, cells[:, ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))].reshape(-1, 3)))
            # A selected cell and its selected surface can share faces. Draw
            # each face once so overlapping selections do not increase opacity.
            _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
            faces = faces[np.sort(unique)]
        self.mesh = None
        self.revision = 0
        device = workspace.model.device
        self.points = wp.empty(len(self.particles), dtype=wp.vec3, device=device)
        self._bounds = wp.empty((2, 3), dtype=float, device=device)
        self.batches = []
        for source in self.catalog.sources:
            start, end = np.searchsorted(self.particles, (source.offset, source.end))
            if end > start:
                indices = wp.array(self.particles[start:end] - source.offset, dtype=int, device=device)
                self.batches.append((source, indices, self.points[start:end]))
        self.empty_offsets = wp.empty(0, dtype=wp.vec3, device=device)
        if len(faces):
            self.faces = wp.array(np.searchsorted(self.particles, faces).ravel(), dtype=int, device=device)
            self.normals = wp.full(len(self.particles), wp.vec3(0, 0, 1), device=device)
            self.mesh = MeshGL(
                len(self.particles),
                self.faces.size,
                device,
                backface_culling=False,
                enable_cuda_interop=self.viewer._cuda_interop_enabled(self.viewer.CudaInterop.DYNAMIC_MESH),
            )
        self.marker_slots = wp.array(np.searchsorted(self.particles, self.markers), dtype=int, device=device)
        self.marker_points = wp.empty(len(self.markers), dtype=wp.vec3, device=device)
        self.marker_colors = wp.full(len(self.markers), wp.vec3(1.0, 0.43, 0.07), device=device)

    def close(self) -> None:
        if self.mesh is not None:
            renderer = self.viewer.renderer
            renderer._selection_instances = tuple(
                (obj, index) for obj, index in renderer._selection_instances if obj is not self.mesh
            )
            self.mesh.destroy()
            self.mesh = None

    def update(self, workspace: ViewerWorkspace) -> None:
        state, viewer, model = workspace.viewer._last_state, self.viewer, workspace.model
        if state is None or not len(self.particles):
            return
        for source, indices, output in self.batches:
            positions = state.particle_q if source is self.catalog.sources[0] else source.points
            if positions is None:
                return
            wp.launch(
                gather_display_particles,
                dim=len(indices),
                inputs=[
                    indices,
                    positions,
                    source.model.particle_world,
                    viewer.world_offsets if viewer.world_offsets is not None else self.empty_offsets,
                    viewer.layer.xform,
                ],
                outputs=[output],
                device=model.device,
            )
        self.revision += 1
        if self.mesh is not None:
            self.mesh.update(self.points, self.faces, self.normals, None)
            viewer.renderer._selection_instances += ((self.mesh, -1),)
        if len(self.markers):
            wp.launch(
                _gather_markers,
                dim=len(self.markers),
                inputs=[self.marker_slots, self.points],
                outputs=[self.marker_points],
                device=model.device,
            )
            radii = viewer.camera.pivot_distance * 0.003
            if viewer.show_particles:
                radii = max(radii, *(source.model.particle_max_radius * 1.05 for source, _, _ in self.batches))
            viewer.log_points(
                "/workspace/selected_particles", self.marker_points, radii=radii, colors=self.marker_colors
            )

    def bounds(self) -> np.ndarray | None:
        if not len(self.particles):
            return None
        bounds = self._bounds
        bounds[0:1].fill_(1e30)
        bounds[1:2].fill_(-1e30)
        wp.launch(
            particle_bounds, dim=len(self.points), inputs=[self.points], outputs=[bounds], device=self.points.device
        )
        return bounds.numpy()


class DeformableDisplay:
    """Apply world spacing and layer transforms without changing simulation positions.

    Display points and radii update on the device each frame. Triangle filtering
    is rebuilt only when appearance groups or visibility change. World offsets
    are indexed by original world ID, independent of which worlds are visible.
    """

    def __init__(self, viewer: ViewerBase) -> None:
        self.model = viewer.model
        self.points = wp.empty(self.model.particle_count, dtype=wp.vec3, device=viewer.device)
        self.radii = wp.empty(self.model.particle_count, dtype=float, device=viewer.device)
        self.empty_offsets = wp.empty(0, dtype=wp.vec3, device=viewer.device)
        self.empty_flags = wp.empty(0, dtype=int, device=viewer.device)
        starts = self.model.particle_world_start.numpy()
        self.worlds = np.searchsorted(starts, np.arange(self.model.particle_count), side="right") - 1
        self.worlds[(self.worlds < 0) | (self.worlds >= self.model.world_count)] = -1
        self.key = None
        self.groups = []
        self.visible = wp.ones(self.model.particle_count, dtype=int, device=viewer.device)

    def update(self, viewer: ViewerBase, state: State) -> list[str]:
        workspace = getattr(viewer.gui, "workspace", None)
        catalog = (
            workspace.tree.deformables
            if workspace is not None and workspace.tree is not None and workspace.tree.model is self.model
            else None
        )
        hidden = tuple(sorted(catalog.hidden)) if catalog is not None else ()
        groups, stale = viewer._get_triangle_appearance_groups() if self.model.tri_count else ([], [])
        key = (
            id(groups) if self.model.tri_count else None,
            id(catalog),
            hidden,
            tuple(viewer._visible_worlds) if viewer._visible_worlds is not None else None,
        )
        if key != self.key:
            visible = np.ones(self.model.particle_count, dtype=bool)
            if viewer._visible_worlds is not None:
                visible &= (self.worlds < 0) | np.isin(self.worlds, tuple(viewer._visible_worlds))
            for obj_key in hidden:
                indices = np.asarray(catalog.objects[obj_key[1]].particles, dtype=int)
                visible[indices[indices < self.model.particle_count]] = False
            self.visible.assign(visible.astype(np.int32))
            self.groups = []
            for name, indices, color, opacity in groups:
                if visible.all():
                    filtered = indices
                else:
                    triangles = indices.numpy().reshape(-1, 3)
                    filtered = wp.array(
                        triangles[visible[triangles].all(axis=1)].ravel(), dtype=int, device=viewer.device
                    )
                self.groups.append((name, filtered, color, opacity))
            self.key = key
        wp.launch(
            _display_all_particles,
            dim=self.model.particle_count,
            inputs=[
                state.particle_q,
                self.model.particle_world,
                viewer.world_offsets if viewer.world_offsets is not None else self.empty_offsets,
                viewer.layer.xform,
                self.model.particle_radius,
                self.model.particle_flags if self.model.particle_flags is not None else self.empty_flags,
                self.visible,
            ],
            outputs=[self.points, self.radii],
            device=viewer.device,
        )
        return stale

    def particle_batch(self) -> tuple[wp.array[wp.vec3], wp.array[float], wp.array[wp.vec3]]:
        # Keep inactive emitter capacity and hidden worlds out of the draw
        # call, matching the regular viewer's compact particle path.
        count, device = len(self.points), self.points.device
        if not hasattr(self, "particle_mask"):
            self.particle_mask = wp.empty(count, dtype=int, device=device)
            self.particle_offsets = wp.empty(count, dtype=int, device=device)
            self.compact_points = wp.empty(count, dtype=wp.vec3, device=device)
            self.compact_radii = wp.empty(count, dtype=float, device=device)
            self.colors = wp.full(count, wp.vec3(0.7, 0.6, 0.4), device=device)
        wp.launch(_particle_mask, dim=count, inputs=[self.radii], outputs=[self.particle_mask], device=device)
        wp.utils.array_scan(self.particle_mask, self.particle_offsets, inclusive=True)
        active = int(self.particle_offsets[count - 1 : count].numpy()[0])
        if active == count:
            return self.points, self.radii, self.colors
        wp.launch(
            _compact_display_particles,
            dim=count,
            inputs=[self.points, self.radii, self.particle_mask, self.particle_offsets],
            outputs=[self.compact_points, self.compact_radii],
            device=device,
        )
        return self.compact_points[:active], self.compact_radii[:active], self.colors[:active]
