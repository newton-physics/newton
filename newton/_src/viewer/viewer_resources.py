# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lazy inspection of shared collision SDFs and mesh texture resources."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .viewer_bvh import BvhResources

if TYPE_CHECKING:
    from typing import Any

    from newton import Gaussian, Heightfield, Mesh, Model
    from newton._src.geometry.sdf_texture import TextureSDFData
    from newton._src.geometry.sdf_utils import SDF
    from newton._src.viewer.gui.workspace import ViewerWorkspace
    from newton._src.viewer.viewer_model_tree import ModelTree
    from newton._src.viewer.viewer_types import TreeKey, ViewerExample

    from .gui.widgets import PhysicsWidgets
    from .viewer_session import ViewerSession


class GeometryResources:
    """Read metadata on demand, without building SDFs or downloading texture pixels."""

    root = ("sdf_collection", 0)
    page_size = 128

    def __init__(
        self, model: Model, example: ViewerExample | None = None, *, session: ViewerSession | None = None
    ) -> None:
        """Create lazy SDF and BVH catalogs without reading texture payloads or SDF rows."""
        self.model = model
        self.bvhs = BvhResources(model, example, session=session)
        data = model._texture_sdf_data
        self.count = len(data) if data is not None else 0
        self._shape_pages = {}
        self._rows = {}
        self._owners = None
        self._users_cache = {}

    @staticmethod
    def handles(key: TreeKey) -> bool:
        """Recognize geometry-resource and BVH keys that this inspector can display."""
        return BvhResources.handles(key) or (
            key is not None
            and key[0]
            in (
                "sdf_collection",
                "sdf_page",
                "sdf",
                "sdf_texture",
                "sdf_slots",
                "shape_sdf",
                "mesh_sdf",
                "shape_texture",
            )
        )

    @staticmethod
    def shape_owner(key: TreeKey) -> int | None:
        """Return the owning shape index for a per-shape resource key, or None otherwise."""
        return key[1] if key[0] in ("shape_sdf", "mesh_sdf", "shape_texture", "shape_bvh") else None

    def shape_sdf(self, index: int) -> int:
        """Resolve a shape to a Model SDF index, returning -1 when no mapping exists.

        Copy only the containing page of shape-to-SDF indices on first access.
        Cached pages assume the Model layout remains unchanged.
        """
        array = self.model._shape_sdf_index
        if array is None or index >= len(array):
            return -1
        page = index // self.page_size
        if page not in self._shape_pages:
            start = page * self.page_size
            self._shape_pages[page] = array[start : min(start + self.page_size, len(array))].numpy()
        return int(self._shape_pages[page][index % self.page_size])

    def source(self, shape: int) -> Mesh | Heightfield | Gaussian | None:
        """Return the Model-owned source geometry for a valid shape index."""
        return self.model.shape_source[shape]

    @staticmethod
    def color_texture_loaded(ws: ViewerWorkspace, shape: int) -> bool:
        """Check whether the shape batch has an uploaded color texture in the GL viewer."""
        batches = ws.viewer._shape_to_batch
        batch = batches[shape] if batches is not None else None
        obj = ws.viewer.objects.get(batch.name) if batch is not None else None
        return getattr(getattr(obj, "mesh", None), "texture_id", None) is not None

    def attach(self, tree: ModelTree) -> None:
        """Attach BVH roots and a shared SDF root, leaving individual resource rows lazy."""
        self.bvhs.attach(tree)
        if self.count:
            tree.roots.append(self.root)
            tree.labels[self.root] = f"SDFs · {self.count:,}"
            tree.meta[self.root] = {"world": -1}

    def _node(self, tree: ModelTree, parent: TreeKey, key: TreeKey, label: str, *, world: int = -1) -> None:
        """Attach or refresh one resource node without duplicating its parent-child link."""
        if key not in tree.parents:
            tree.children[parent].append(key)
        tree.parents[key] = parent
        tree.labels[key] = label
        tree.meta[key] = {"world": world}

    def add_shape(self, tree: ModelTree, shape: int) -> None:
        """Attach available BVH, mesh SDF, Model SDF, and color-texture references beneath a shape."""
        self.bvhs.add_shape(tree, shape)
        parent = ("shape", shape)
        source, index = self.source(shape), self.shape_sdf(shape)
        world = tree.world_for(parent)
        if 0 <= index < self.count:
            self._node(tree, parent, ("shape_sdf", shape), f"Model SDF {index}", world=world)
        if getattr(source, "sdf", None) is not None:
            self._node(tree, parent, ("mesh_sdf", shape), "Mesh SDF", world=world)
        if getattr(source, "texture", None) is not None:
            self._node(tree, parent, ("shape_texture", shape), "Color texture · 2D", world=world)

    def has_children(self, tree: ModelTree, key: TreeKey) -> bool:
        """Determine expandability, materializing shape or SDF resource children when needed."""
        if self.bvhs.handles(key):
            return key[0] in ("bvh_collection", "bvh_page") or bool(tree.children.get(key))
        if key[0] == "shape":
            self.add_shape(tree, key[1])
        elif key[0] in ("sdf_collection", "sdf_page"):
            return True
        elif key[0] == "sdf":
            self.expand(tree, key)
        return bool(tree.children.get(key))

    def textures(self, index: int) -> tuple[wp.Texture3D | None, wp.Texture3D | None, wp.array[int] | None]:
        """Return coarse texture, subgrid texture, and indirection array references for an SDF."""
        model = self.model
        return (
            model._texture_sdf_coarse_textures[index] if index < len(model._texture_sdf_coarse_textures) else None,
            model._texture_sdf_subgrid_textures[index] if index < len(model._texture_sdf_subgrid_textures) else None,
            model._texture_sdf_subgrid_start_slots[index]
            if index < len(model._texture_sdf_subgrid_start_slots)
            else None,
        )

    def expand(self, tree: ModelTree, key: TreeKey) -> None:
        """Populate one bounded page or an SDF entry, reusing already materialized children."""
        if self.bvhs.handles(key):
            self.bvhs.expand(tree, key)
            return
        if tree.children.get(key):
            return
        if key[0] in ("sdf_collection", "sdf_page"):
            first, end = (0, self.count) if key == self.root else key[1]
            if end - first > self.page_size:
                size = self.page_size
                while end - first > size * self.page_size:
                    size *= self.page_size
                for start in range(first, end, size):
                    stop = min(start + size, end)
                    self._node(tree, key, ("sdf_page", (start, stop)), f"SDFs {start:,}-{stop - 1:,}")
            else:
                for index in range(first, end):
                    self._node(tree, key, ("sdf", index), f"SDF {index}")
        elif key[0] == "sdf":
            coarse, subgrid, slots = self.textures(key[1])
            for name, texture in (("Coarse", coarse), ("Subgrid", subgrid)):
                if texture is not None:
                    self._node(tree, key, ("sdf_texture", (key[1], name)), f"{name} · 3D texture")
            if slots is not None:
                self._node(tree, key, ("sdf_slots", key[1]), "Subgrid indirection")

    def reveal(self, tree: ModelTree, key: TreeKey) -> None:
        """Materialize the page chain or owning shape needed to navigate to a resource key."""
        if self.bvhs.handles(key):
            self.bvhs.reveal(tree, key)
            return
        if key in tree.labels:
            return
        shape = self.shape_owner(key)
        if shape is not None:
            tree.ensure_world(tree.world_for(("shape", shape)))
            self.add_shape(tree, shape)
            return
        index = key[1][0] if key[0] == "sdf_texture" else key[1]
        if not isinstance(index, int) or not 0 <= index < self.count:
            return
        parent = self.root
        while ("sdf", index) not in tree.labels:
            self.expand(tree, parent)
            parent = next(
                (
                    child
                    for child in tree.children[parent]
                    if child[0] == "sdf_page" and child[1][0] <= index < child[1][1]
                ),
                ("sdf", index),
            )
        self.expand(tree, ("sdf", index))

    def row(self, index: int) -> np.void:
        """Copy one SDF descriptor to the host on first access and return its cached row."""
        if index not in self._rows:
            self._rows[index] = self.model._texture_sdf_data[index : index + 1].numpy()[0]
        return self._rows[index]

    @staticmethod
    def sdf_storage(sdf: SDF | None) -> str:
        """Describe whether an SDF is absent, empty, texture-backed, or NanoVDB-backed."""
        if sdf is None:
            return "Not built"
        if sdf.is_empty():
            return "Empty"
        return "3D textures" if sdf.texture_data is not None else "NanoVDB volumes"

    def shape_summary(self, ws: ViewerWorkspace, shape: int) -> None:
        """Draw a shape geometry summary with navigation to its Model SDF and texture status."""
        w, source, index = ws.widgets, self.source(shape), self.shape_sdf(shape)
        with w.group("Geometry resources") as opened:
            if not opened:
                return
            vertices, indices = getattr(source, "vertices", None), getattr(source, "indices", None)
            if vertices is not None:
                w.number("Vertices", len(vertices), integer=True)
            if indices is not None:
                w.number("Triangles", np.size(indices) // 3, integer=True)
            if hasattr(source, "sdf"):
                w.text_value("Mesh SDF", self.sdf_storage(source.sdf))
            target = f"SDF {index}" if 0 <= index < self.count else None
            if w.reference("Model SDF", index, target=target):
                ws.navigate(("sdf", index))
            texture = getattr(source, "texture", None)
            status = "Loaded" if self.color_texture_loaded(ws, shape) else "Assigned"
            w.text_value("Color texture", f"{status} · 2D image" if texture is not None else "None")

    @staticmethod
    def _texture_metadata(w: PhysicsWidgets, texture: wp.Texture3D | None) -> None:
        """Display texture allocation metadata and estimated texel bytes without reading texels."""
        if texture is None:
            w.text_value("Status", "Not built")
            return
        w.text_value("Status", "Allocated")
        w.array("Dimensions", (texture.width, texture.height, texture.depth), labels=("X", "Y", "Z"), integer=True)
        w.text_value("Dtype", wp.types.type_repr(texture.dtype))
        w.number("Channels", texture.num_channels, integer=True)
        w.text_value("Device", str(texture.device))
        w.text_value("Filtering", texture.filter_mode.name.title())
        w.boolean("Normalized coordinates", texture.normalized_coords)
        size = (
            texture.width
            * texture.height
            * texture.depth
            * texture.num_channels
            * wp.types.type_size_in_bytes(texture.dtype)
        )
        w.text_value("Texel payload", f"{size / 1048576:.3f} MiB")

    @staticmethod
    def _array_metadata(w: PhysicsWidgets, array: wp.array[Any] | None) -> None:
        """Display shape, dtype, device, and payload size for an existing Warp array."""
        w.text_value("Dimensions", " x ".join(str(n) for n in array.shape))
        w.text_value("Dtype", wp.types.type_repr(array.dtype))
        w.text_value("Device", str(array.device))
        w.text_value("Storage", f"{array.size * wp.types.type_size_in_bytes(array.dtype) / 1048576:.3f} MiB")

    @staticmethod
    def _sdf_metadata(w: PhysicsWidgets, data: np.void | TextureSDFData) -> None:
        """Display sampling, local bounds, and distance encoding from a host SDF descriptor."""

        def field(name: str) -> Any:
            return data[name] if isinstance(data, np.void) else getattr(data, name)

        with w.group("Sampling") as opened:
            if opened:
                w.vec3("Voxel size", field("voxel_size"), unit="m")
                w.number("Voxel radius", float(field("voxel_radius")), unit="m")
                w.number("Subgrid size", int(field("subgrid_size")), integer=True)
                w.number("Subgrids", int(field("num_subgrids")), integer=True)
                w.boolean("Paired X samples", bool(field("paired_samples")))
                w.boolean("Scale baked", bool(field("scale_baked")))
        with w.group("Local bounds") as opened:
            if opened:
                w.vec3("Lower", field("sdf_box_lower"), unit="m")
                w.vec3("Upper", field("sdf_box_upper"), unit="m")
        with w.group("Distance encoding", default_open=False) as opened:
            if opened:
                w.number("Minimum", float(field("subgrids_min_sdf_value")), unit="m")
                w.number("Range", float(field("subgrids_sdf_value_range")), unit="m")

    def _users(self, ws: ViewerWorkspace, index: int) -> None:
        """Show paged shape references for an SDF when its usage group is open.

        The first request copies the shape-to-SDF map to the host. Cache inverse
        references for subsequent pages without repeatedly reading the device.
        """
        with ws.widgets.group("Used by shapes", default_open=False) as opened:
            if not opened:
                return
            # One index array read only when the user requests inverse references.
            if self._owners is None:
                array = self.model._shape_sdf_index
                self._owners = array.numpy() if array is not None else np.empty(0, dtype=int)
            if index not in self._users_cache:
                self._users_cache[index] = np.flatnonzero(self._owners == index)
            owners = self._users_cache[index]
            ws.widgets.number("Shapes", len(owners), integer=True)
            for entry in ws.property_inspector._page(owners):
                shape = int(entry)
                label = self.model.shape_label[shape] or f"Shape {shape}"
                key = ("shape", shape)
                caption = ws.tree.reference_caption(key, label)
                if ws.widgets.reference(f"Shape {shape}", shape, target=label, caption=caption):
                    ws.navigate(key)

    def draw(self, ws: ViewerWorkspace) -> None:
        """Render the selected geometry-resource page, delegating BVH properties to its catalog."""
        if self.bvhs.handles(ws.selected):
            self.bvhs.draw(ws)
            return
        w, ui = ws.widgets, ws.gui.ui.imgui
        key = ws.selected
        kind, index = key
        if not ui.begin_tab_bar("##resource_properties"):
            return
        if w.tab_item("Model")[0]:
            if kind in ("sdf_collection", "sdf_page"):
                w.number("SDFs", self.count if kind == "sdf_collection" else index[1] - index[0], integer=True)
                w.text_value("Storage", "Shared Model table")
                w.boolean("Paired X samples", self.model._sdf_texture_paired_samples)
            elif kind in ("sdf", "shape_sdf"):
                if kind == "shape_sdf":
                    index = self.shape_sdf(index)
                    if w.reference("Model SDF", index, target=f"SDF {index}"):
                        ws.navigate(("sdf", index))
                coarse, subgrid, slots = self.textures(index)
                with w.group("SDF") as opened:
                    if opened:
                        w.number("Index", index, integer=True)
                        w.text_value("Status", "Built" if coarse is not None else "Empty")
                        w.text_value("Storage", "3D textures" if coarse is not None else "Empty texture entry")
                        w.text_value("Device", str(self.model._texture_sdf_data.device))
                        if subgrid is not None:
                            w.text_value("Subgrid format", wp.types.type_repr(subgrid.dtype))
                self._sdf_metadata(w, self.row(index))
                for name, texture in (("Coarse", coarse), ("Subgrid", subgrid)):
                    with w.group(name + " texture", default_open=False) as opened:
                        if opened:
                            self._texture_metadata(w, texture)
                if slots is not None:
                    with w.group("Subgrid indirection", default_open=False) as opened:
                        if opened:
                            self._array_metadata(w, slots)
                self._users(ws, index)
            elif kind == "sdf_texture":
                self._texture_metadata(w, self.textures(index[0])[0 if index[1] == "Coarse" else 1])
            elif kind == "sdf_slots":
                self._array_metadata(w, self.textures(index)[2])
            elif kind == "mesh_sdf":
                sdf = self.source(index).sdf
                w.text_value("Storage", self.sdf_storage(sdf))
                w.number("Shape margin", sdf.shape_margin, unit="m")
                if sdf.texture_data is not None:
                    self._sdf_metadata(w, sdf.texture_data)
                    for name, texture in (("Coarse", sdf._coarse_texture), ("Subgrid", sdf._subgrid_texture)):
                        with w.group(name + " texture", default_open=False) as opened:
                            if opened:
                                self._texture_metadata(w, texture)
                elif not sdf.is_empty():
                    w.vec3("Center", sdf.data.center, unit="m")
                    w.vec3("Half extents", sdf.data.half_extents, unit="m")
                    w.vec3("Sparse voxel size", sdf.data.sparse_voxel_size, unit="m")
                    w.vec3("Coarse voxel size", sdf.data.coarse_voxel_size, unit="m")
                    w.boolean("Scale baked", sdf.data.scale_baked)
            elif kind == "shape_texture":
                source = self.source(index)
                texture = source.texture
                w.text_value("Use", "Surface color (2D image)")
                loaded = self.color_texture_loaded(ws, index)
                w.text_value("Viewer upload", "Loaded" if loaded else "Not loaded")
                if isinstance(texture, np.ndarray):
                    w.text_value("Dimensions", " x ".join(str(n) for n in texture.shape))
                    w.text_value("Dtype", str(texture.dtype))
                else:
                    w.text_value("Source", str(texture))
                w.boolean("UV coordinates", getattr(source, "uvs", None) is not None)
                w.array("UV transform", source.texture_transform)
            ui.end_tab_item()
        ui.end_tab_bar()
