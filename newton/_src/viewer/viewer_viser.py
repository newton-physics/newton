# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import inspect
import os
import queue
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import numpy as np
import warp as wp

import newton

from ..core.types import Axis, override
from ..utils.texture import load_texture, normalize_texture
from .picking import Picking
from .viewer import ViewerBase, is_jupyter_notebook


class ViewerViser(ViewerBase):
    """
    ViewerViser provides a backend for visualizing Newton simulations using the viser library.

    Viser is a Python library for interactive 3D visualization in the browser. This viewer
    launches a web server and renders simulation geometry via WebGL. It supports both
    standalone browser viewing and Jupyter notebook integration.

    Features:
        - Real-time 3D visualization in any web browser
        - Jupyter notebook integration with inline display
        - Static HTML export for sharing visualizations
        - Interactive camera controls
        - Interactive transform gizmos and live scalar plots
        - Picking forces through click-to-create translation handles

    Click a rendered body to create a picking handle at the hit point. Drag
    the handle to apply a spring force; releasing the handle drops the body.
    A cyan line connects the moving body anchor to the handle target.
    """

    _viser_module = None

    @property
    def picking_enabled(self) -> bool:
        """Whether rendered bodies can be selected for force picking."""
        return self._picking_enabled

    @picking_enabled.setter
    def picking_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        was_enabled = getattr(self, "_picking_enabled", False)
        object.__setattr__(self, "_picking_enabled", enabled)

        callbacks = getattr(self, "_picking_click_callbacks", None)
        if callbacks is None or enabled == was_enabled:
            return
        if not enabled:
            for handle, _callback in list(callbacks.values()):
                self._detach_picking_callback(handle)
            for layer_id in list(getattr(self, "_picking_controls", {})):
                self._remove_picking_control(layer_id, release=True)
            return

        for name, instance in getattr(self, "_instances", {}).items():
            handle = getattr(self, "_scene_handles", {}).get(name)
            if handle is not None and instance.get("pickable", False):
                self._attach_picking_callback(handle, instance["layer_id"])

    @classmethod
    def _get_viser(cls):
        """Lazily import and return the viser module."""
        if cls._viser_module is None:
            try:
                import viser

                cls._viser_module = viser
            except ImportError as e:
                raise ImportError("viser package is required for ViewerViser. Install with: pip install viser") from e
        return cls._viser_module

    @staticmethod
    def _to_numpy(x) -> np.ndarray | None:
        """Convert warp arrays or other array-like objects to numpy arrays."""
        if x is None:
            return None
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    @staticmethod
    def _prepare_texture(texture: np.ndarray | str | None) -> np.ndarray | None:
        """Load and normalize texture data for viser/glTF usage."""
        return normalize_texture(
            load_texture(texture),
            flip_vertical=False,
            require_channels=True,
            scale_unit_range=True,
        )

    @staticmethod
    def _build_trimesh_mesh(points: np.ndarray, indices: np.ndarray, uvs: np.ndarray, texture: np.ndarray):
        """Create a trimesh object with texture visuals (if trimesh is available)."""
        try:
            import trimesh
        except Exception:
            return None

        if len(uvs) != len(points):
            return None

        faces = indices.astype(np.int64)
        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

        try:
            from PIL import Image
            from trimesh.visual.texture import TextureVisuals

            image = Image.fromarray(texture)
            mesh.visual = TextureVisuals(uv=uvs, image=image)
        except Exception:
            visual_mod = getattr(trimesh, "visual", None)
            TextureVisuals = getattr(visual_mod, "TextureVisuals", None) if visual_mod is not None else None
            if TextureVisuals is not None:
                mesh.visual = TextureVisuals(uv=uvs, image=texture)

        return mesh

    def __init__(
        self,
        *,
        port: int = 8080,
        label: str | None = None,
        verbose: bool = True,
        share: bool = False,
        record_to_viser: str | None = None,
        plot_history_size: int = 250,
    ):
        """
        Initialize the ViewerViser backend for Newton using the viser visualization library.

        This viewer supports both standalone browser viewing and Jupyter notebook environments.
        It launches a web server that serves an interactive 3D visualization.

        Args:
            port: Port number for the web server. Defaults to 8080.
            label: Optional label for the viser server window title.
            verbose: If True, print the server URL when starting. Defaults to True.
            share: If True, create a publicly accessible URL via viser's share feature.
            record_to_viser: Path to record the viewer to a ``*.viser`` recording file
                (e.g. "my_recording.viser"). If None, the viewer will not record to a file.
            plot_history_size: Maximum number of samples kept per
                :meth:`log_scalar` signal for the live time-series plots.
        """
        if not isinstance(plot_history_size, int) or isinstance(plot_history_size, bool):
            raise TypeError("plot_history_size must be an integer")
        if plot_history_size <= 0:
            raise ValueError("plot_history_size must be > 0")

        viser = self._get_viser()

        # Rolling buffers for log_scalar() time-series plots.
        self._scalar_buffers: dict[str, collections.deque] = {}
        self._scalar_accumulators: dict[str, list[float]] = {}
        self._scalar_smoothing: dict[str, int] = {}
        self._scalar_dirty: set[str] = set()
        self._plot_handles: dict[str, Any] = {}
        self._plot_folder: Any = None
        self._plot_history_size = plot_history_size
        self._plane_meshes = {}
        self._plane_handles = {}
        self._gizmo_handles: dict[str, dict[str, Any]] = {}
        self._gizmo_seen: set[str] = set()
        self._active_gizmos: set[str] = set()
        self.gizmo_is_using = False
        self._picking_controls: dict[str, Any] = {}
        self._picking_click_callbacks: dict[int, tuple[Any, Any]] = {}
        self._interaction_events: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self._paused = False
        self._step_requested = False
        self._reset_callback: Callable[[], None] | None = None
        self._wireframe = False
        self._viewer_option_handles: dict[str, Any] = {}
        self._simulation_gui_handles: dict[str, Any] = {}
        self._example_browser_folder: Any = None
        self._example_browser_handles: dict[str, Any] = {}
        self._example_browser_options: dict[str, str] = {}
        self._example_switch_callback: Callable[[str], None] | None = None
        self._frame_atomic_context: Any = None

        super().__init__()

        self._running = True
        self.verbose = verbose

        # Store mesh data for instances
        self._meshes = {}
        self._instances = {}
        self._scene_handles = {}  # Track viser scene node handles
        self._line_segment_counts = {}
        self._line_versions = {}

        # Initialize viser server
        self._server = viser.ViserServer(port=port, label=label or "Newton Viewer")
        self._camera_request: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._pending_camera_clients: set[int] = set()
        self._server.on_client_connect(self._handle_client_connect)
        self._server.on_client_disconnect(self._handle_client_disconnect)

        # Store configuration before any URL generation.
        self._port = port
        self.is_jupyter_notebook = is_jupyter_notebook()

        if share:
            self._share_url = self._server.request_share_url()
            if verbose:
                print(f"Viser share URL: {self._share_url}")
        else:
            self._share_url = None

        if verbose:
            print(f"Viser server running at: {self.url}")

        # Recording state
        self._frame_dt = 0.0
        self._record_to_viser = record_to_viser
        self._serializer = self._server.get_scene_serializer() if record_to_viser else None

        # Set up default scene
        self._setup_scene()
        self._setup_gui()

        if self._serializer is not None and verbose:
            print(f"Recording to: {record_to_viser}")

    @override
    def clear_model(self):
        """Reset model-dependent state, including scalar plot buffers."""
        owns = self._is_layer_owned_path

        self._remove_picking_control(self.layer.layer_id, release=True)
        for name in list(self._gizmo_handles):
            if owns(name):
                self._remove_gizmo(name)

        for plane_name in list(self._plane_handles.keys()):
            if owns(plane_name):
                self._remove_plane_handles(plane_name)
        self._plane_meshes = {name: value for name, value in self._plane_meshes.items() if not owns(name)}

        for name, handle in list(getattr(self, "_scene_handles", {}).items()):
            if not owns(name):
                continue
            self._detach_picking_callback(handle)
            try:
                handle.remove()
            except Exception:
                pass
            self._scene_handles.pop(name, None)
            self._instances.pop(name, None)
            self._meshes.pop(name, None)
            self._line_segment_counts.pop(name, None)
            self._line_versions.pop(name, None)

        # Remove only scalar plot state owned by the active layer.
        for name, handle in list(self._plot_handles.items()):
            if not owns(name):
                continue
            try:
                handle.remove()
            except Exception:
                pass
            self._plot_handles.pop(name, None)
        if not self._plot_handles and self._plot_folder is not None:
            try:
                self._plot_folder.remove()
            except Exception:
                pass
            self._plot_folder = None
        for name in list(self._scalar_buffers.keys()):
            if owns(name):
                self._scalar_buffers.pop(name, None)
                self._scalar_accumulators.pop(name, None)
                self._scalar_smoothing.pop(name, None)
        for name in list(self._scalar_dirty):
            if owns(name):
                self._scalar_dirty.discard(name)

        super().clear_model()
        self._sync_gui_controls()

    @override
    def _init_extra_layer_state(self, layer):
        """Initialize Viser-specific per-layer interaction state."""
        super()._init_extra_layer_state(layer)
        layer.picking = None
        layer._last_state = None

    @override
    def set_model(self, model: newton.Model | None):
        """Set the model and initialize interactive picking for it.

        Args:
            model: Model to visualize and interact with.
        """
        super().set_model(model)
        self.picking = Picking(model, world_offsets=self.world_offsets) if model is not None else None
        if self.picking is not None:
            self.picking.visible_worlds_mask = self._visible_worlds_mask

        if model is not None:
            try:
                from ..geometry import raycast as _raycast_module  # noqa: PLC0415

                wp.load_module(module=_raycast_module, device=model.device)
                wp.load_module(module="newton._src.viewer.kernels", device=model.device)
            except Exception:
                pass

    @override
    def set_visible_worlds(self, worlds: Sequence[int] | None) -> None:
        """Set visible worlds and keep picking visibility in sync.

        Args:
            worlds: World indices to show, or ``None`` to show all worlds.
        """
        super().set_visible_worlds(worlds)
        if self.picking is not None:
            self.picking.visible_worlds_mask = self._visible_worlds_mask

    @override
    def set_world_offsets(self, spacing: tuple[float, float, float] | list[float] | wp.vec3) -> None:
        """Set world offsets and keep picking coordinates in sync.

        Args:
            spacing: Spacing between worlds along each axis [m].
        """
        super().set_world_offsets(spacing)
        if self.picking is not None:
            self.picking.world_offsets = self.world_offsets

    def _setup_scene(self):
        """Set up the default scene configuration."""

        self._server.scene.add_light_ambient("ambient_light")

        # remove HDR map
        self._server.scene.configure_environment_map(hdri=None)

    def _begin_atomic_frame(self) -> None:
        """Hold outgoing scene messages until the complete frame is ready."""
        self._end_atomic_frame()
        self._frame_atomic_context = self._server.atomic()
        self._frame_atomic_context.__enter__()

    def _end_atomic_frame(self) -> None:
        """Publish a pending frame as one atomic Viser message group."""
        context = self._frame_atomic_context
        if context is None:
            return
        self._frame_atomic_context = None
        context.__exit__(None, None, None)

    def _setup_gui(self) -> None:
        """Create native Viser controls for supported viewer settings."""
        with self._server.gui.add_folder("Simulation", order=0.0):
            pause = self._server.gui.add_checkbox("Pause", initial_value=self._paused)
            step = self._server.gui.add_button("Step", disabled=True)
            reset = self._server.gui.add_button("Reset", disabled=True)

        @pause.on_update
        def _on_pause(event):
            if event.client_id is not None:
                self._interaction_events.put(("viewer_option", "_paused", bool(event.target.value)))

        @step.on_click
        def _on_step(_event):
            self._interaction_events.put(("step",))

        @reset.on_click
        def _on_reset(_event):
            self._interaction_events.put(("reset",))

        self._simulation_gui_handles = {"pause": pause, "step": step, "reset": reset}

        options = (
            ("show_visual", "Show Visual", None),
            ("show_collision", "Show Collision", None),
            (
                "wireframe",
                "Wireframe (untextured)",
                "Viser exposes wireframe rendering for untextured meshes only.",
            ),
            ("show_ground", "Show Ground", None),
            ("show_contacts", "Show Contacts", "Requires the example to log contacts."),
            ("show_contact_normals", "Contact Normals", None),
            ("show_contact_disks", "Contact Modes", None),
            ("show_contact_forces", "Contact Forces", None),
            ("show_joints", "Show Joints", None),
            ("show_com", "Show Center of Mass", None),
            ("show_particles", "Show Particles", None),
            ("show_springs", "Show Springs", None),
            ("show_triangles", "Show Cloth", None),
            ("show_inertia_boxes", "Show Inertia Boxes", None),
            ("picking_enabled", "Enable Picking", None),
        )
        with self._server.gui.add_folder("Visualization", order=10.0):
            for attribute, label, hint in options:
                value = self._wireframe if attribute == "wireframe" else bool(getattr(self, attribute))
                handle = self._server.gui.add_checkbox(label, initial_value=value, hint=hint)

                @handle.on_update
                def _on_option(event, option=attribute):
                    if event.client_id is not None:
                        self._interaction_events.put(("viewer_option", option, bool(event.target.value)))

                self._viewer_option_handles[attribute] = handle

    def _sync_gui_controls(self) -> None:
        """Synchronize native GUI values with viewer state."""
        for attribute, handle in getattr(self, "_viewer_option_handles", {}).items():
            value = self._wireframe if attribute == "wireframe" else bool(getattr(self, attribute))
            if handle.value != value:
                handle.value = value

        simulation = getattr(self, "_simulation_gui_handles", {})
        if simulation:
            if simulation["pause"].value != self._paused:
                simulation["pause"].value = self._paused
            step_disabled = not self._paused
            if simulation["step"].disabled != step_disabled:
                simulation["step"].disabled = step_disabled
            reset_disabled = self._reset_callback is None
            if simulation["reset"].disabled != reset_disabled:
                simulation["reset"].disabled = reset_disabled

    def _set_wireframe(self, enabled: bool) -> None:
        """Set wireframe rendering on Viser mesh batches that support it."""
        self._wireframe = bool(enabled)
        for name, handle in self._scene_handles.items():
            instance = self._instances.get(name)
            if instance is None or instance["use_trimesh"]:
                continue
            try:
                handle.wireframe = self._wireframe
            except Exception:
                pass

    def configure_example_browser(
        self,
        tree: dict[str, list[tuple[str, str]]],
        callback: Callable[[str], None],
    ) -> None:
        """Configure a native dropdown for switching Newton examples.

        Args:
            tree: Example names and module paths grouped by category.
            callback: Function called with the selected module path.
        """
        if self._example_browser_folder is not None:
            try:
                self._example_browser_folder.remove()
            except Exception:
                pass

        options = {
            f"{category} / {name}": module_path
            for category, examples in sorted(tree.items())
            for name, module_path in examples
        }
        self._example_browser_options = options
        self._example_switch_callback = callback
        if not options:
            self._example_browser_folder = None
            self._example_browser_handles = {}
            return

        folder = self._server.gui.add_folder("Examples", order=20.0, expand_by_default=False)
        with folder:
            dropdown = self._server.gui.add_dropdown("Example", tuple(options), initial_value=next(iter(options)))
            load = self._server.gui.add_button("Load Example")

        @load.on_click
        def _on_load(_event):
            module_path = self._example_browser_options.get(dropdown.value)
            if module_path is not None:
                self._interaction_events.put(("example_switch", module_path))

        self._example_browser_folder = folder
        self._example_browser_handles = {"dropdown": dropdown, "load": load}

    def set_reset_callback(self, callback: Callable[[], None] | None) -> None:
        """Register the callback invoked by the native Reset button."""
        self._reset_callback = callback
        self._sync_gui_controls()

    @staticmethod
    def _call_scene_method(method, **kwargs):
        """Call a viser scene method with only supported keyword args."""
        try:
            signature = inspect.signature(method)
            allowed = {k: v for k, v in kwargs.items() if k in signature.parameters}
            return method(**allowed)
        except Exception:
            return method(**kwargs)

    @staticmethod
    def _set_handle_property_if_changed(handle: Any, name: str, value: Any) -> None:
        """Update a Viser property only when its serialized value changed."""
        try:
            current = getattr(handle, name)
            if isinstance(current, (np.ndarray, tuple, list)) or isinstance(value, (np.ndarray, tuple, list)):
                unchanged = np.array_equal(np.asarray(current), np.asarray(value))
            else:
                unchanged = current == value
            if unchanged:
                return
        except Exception:
            pass
        setattr(handle, name, value)

    @staticmethod
    def _transform_to_viser(transform: wp.transform) -> tuple[np.ndarray, np.ndarray]:
        """Return a Warp transform as Viser position and WXYZ rotation."""
        position = np.asarray((transform.p[0], transform.p[1], transform.p[2]), dtype=np.float32)
        wxyz = np.asarray((transform.q[3], transform.q[0], transform.q[1], transform.q[2]), dtype=np.float32)
        return position, wxyz

    @staticmethod
    def _assign_transform(transform: wp.transform, position, wxyz) -> None:
        """Mutate a pass-by-reference Warp transform from Viser values."""
        transform[:] = wp.transform(
            wp.vec3(float(position[0]), float(position[1]), float(position[2])),
            wp.quat(float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])),
        )

    @staticmethod
    def _normalize_gizmo_axes(axes: Sequence[Axis] | None) -> tuple[Axis, ...]:
        """Normalize an optional axis sequence into stable XYZ order."""
        axis_order = (Axis.X, Axis.Y, Axis.Z)
        if axes is None:
            return axis_order
        enabled = {Axis.from_any(axis) for axis in axes}
        return tuple(axis for axis in axis_order if axis in enabled)

    @staticmethod
    def _gizmo_scene_path(name: str, kind: str) -> str:
        """Build an internal scene path for a user-named gizmo handle."""
        return f"/__newton/gizmos/{name.lstrip('/')}/{kind}"

    def _sync_gizmo_handles(self, entry: dict[str, Any]) -> None:
        """Push a gizmo's current pass-by-reference transform to Viser."""
        position, wxyz = self._transform_to_viser(entry["transform"])
        for handle in entry["handles"].values():
            handle.position = position
            handle.wxyz = wxyz

    def _remove_gizmo(self, name: str) -> None:
        """Remove all Viser controls associated with one gizmo."""
        entry = self._gizmo_handles.pop(name, None)
        if entry is not None:
            for handle in entry["handles"].values():
                try:
                    handle.remove()
                except Exception:
                    pass
        self._active_gizmos.discard(name)
        self.gizmo_is_using = bool(self._active_gizmos)

    def _create_gizmo(
        self,
        name: str,
        transform: wp.transform,
        translate: tuple[Axis, ...],
        rotate: tuple[Axis, ...],
        snap_to: wp.transform | None,
    ) -> dict[str, Any]:
        """Create translation and rotation controls for one logged gizmo."""
        position, wxyz = self._transform_to_viser(transform)
        scale = max(float(self.scene_scale) * 1.5, 0.2)
        handles: dict[str, Any] = {}

        def add_handle(kind: str, axes: tuple[Axis, ...], **kwargs):
            active_axes = tuple(axis in axes for axis in (Axis.X, Axis.Y, Axis.Z))
            handle = self._server.scene.add_transform_controls(
                name=self._gizmo_scene_path(name, kind),
                scale=scale,
                active_axes=active_axes,
                depth_test=False,
                position=position,
                wxyz=wxyz,
                **kwargs,
            )

            @handle.on_update
            def _on_update(event, gizmo_name=name):
                self._interaction_events.put(
                    (
                        "gizmo_update",
                        gizmo_name,
                        tuple(float(v) for v in event.target.position),
                        tuple(float(v) for v in event.target.wxyz),
                    )
                )

            @handle.on_drag_start
            def _on_drag_start(_event, gizmo_name=name):
                # Mark active immediately so the simulation thread cannot push
                # its previous transform over an in-flight browser drag.
                self._active_gizmos.add(gizmo_name)
                self.gizmo_is_using = True
                self._interaction_events.put(("gizmo_drag_start", gizmo_name))

            @handle.on_drag_end
            def _on_drag_end(_event, gizmo_name=name):
                self._interaction_events.put(("gizmo_drag_end", gizmo_name))

            handles[kind] = handle

        if translate:
            add_handle("translate", translate, disable_rotations=True)
        if rotate:
            add_handle("rotate", rotate, disable_axes=True, disable_sliders=True)

        entry = {
            "transform": transform,
            "snap_to": snap_to,
            "translate": translate,
            "rotate": rotate,
            "handles": handles,
        }
        self._gizmo_handles[name] = entry
        return entry

    def _attach_picking_callback(self, handle: Any, layer_id: str) -> None:
        """Make a rendered instance batch start picking when clicked."""
        handle_id = id(handle)
        if not self.picking_enabled or handle_id in self._picking_click_callbacks or not hasattr(handle, "on_click"):
            return

        def _on_click(event, clicked_layer_id=layer_id):
            if not self.picking_enabled:
                return
            self._interaction_events.put(
                (
                    "picking_click",
                    clicked_layer_id,
                    tuple(float(v) for v in event.ray_origin),
                    tuple(float(v) for v in event.ray_direction),
                )
            )

        callback = handle.on_click(_on_click)
        self._picking_click_callbacks[handle_id] = (handle, callback)

    def _shape_instance_batch(self, name: str):
        """Return the model shape batch associated with a scene path, if any."""
        return next((batch for batch in self._shape_instances.values() if batch.name == name), None)

    def _is_pickable_instance(self, name: str) -> bool:
        """Return whether an instance batch contains body-attached model shapes."""
        batch = self._shape_instance_batch(name)
        return batch is not None and not batch.static

    def _detach_picking_callback(self, handle: Any) -> None:
        """Remove picking behavior and Viser's clickable hover highlight."""
        entry = self._picking_click_callbacks.pop(id(handle), None)
        if entry is None or not hasattr(handle, "remove_click_callback"):
            return
        try:
            handle.remove_click_callback(entry[1])
        except Exception:
            pass

    @staticmethod
    def _transform_ray_to_layer(layer, ray_origin, ray_direction) -> tuple[wp.vec3, wp.vec3]:
        """Transform a visual-space ray into a layer's pre-transform space."""
        layer_inv = wp.transform_inverse(layer.xform)
        origin = wp.transform_point(
            layer_inv,
            wp.vec3(float(ray_origin[0]), float(ray_origin[1]), float(ray_origin[2])),
        )
        direction = wp.quat_rotate(
            layer_inv.q,
            wp.vec3(float(ray_direction[0]), float(ray_direction[1]), float(ray_direction[2])),
        )
        return origin, direction

    @staticmethod
    def _picking_world_offset(layer, picking: Picking) -> np.ndarray:
        """Return the visual world offset for the currently picked body."""
        offset = np.zeros(3, dtype=np.float32)
        body_idx = int(picking.pick_body.numpy()[0])
        if body_idx < 0 or layer.world_offsets is None or layer.world_offsets.shape[0] == 0:
            return offset
        if layer.model is None or layer.model.body_world is None:
            return offset
        world_idx = int(layer.model.body_world.numpy()[body_idx])
        if 0 <= world_idx < layer.world_offsets.shape[0]:
            offset[:] = layer.world_offsets.numpy()[world_idx]
        return offset

    @classmethod
    def _picking_point_to_visual(cls, layer, picking: Picking, point) -> np.ndarray:
        """Convert a physics-space picking point into Viser scene space."""
        point_with_offset = np.asarray(point, dtype=np.float32) + cls._picking_world_offset(layer, picking)
        visual = wp.transform_point(
            layer.xform,
            wp.vec3(float(point_with_offset[0]), float(point_with_offset[1]), float(point_with_offset[2])),
        )
        return np.asarray((visual[0], visual[1], visual[2]), dtype=np.float32)

    def _picking_control_path(self, layer_id: str) -> str:
        """Build a stable internal path for a layer's picking handle."""
        segment = "default" if not self._layers[layer_id].name_prefix else "_".join(layer_id.split()).replace("/", "_")
        return f"/__newton/picking/{segment}"

    def _remove_picking_control(self, layer_id: str, *, release: bool) -> None:
        """Remove one layer's picking handle and optionally release its body."""
        handle = self._picking_controls.pop(layer_id, None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
        layer = self._layers.get(layer_id)
        picking = getattr(layer, "picking", None) if layer is not None else None
        if release and picking is not None:
            picking.release()

    def _create_picking_control(self, layer_id: str) -> None:
        """Create a translation-only handle at the current picking target."""
        layer = self._layers[layer_id]
        picking = layer.picking
        pick_state = picking.pick_state.numpy()
        target = self._picking_point_to_visual(layer, picking, pick_state[0]["picking_target_world"])
        self._remove_picking_control(layer_id, release=False)
        handle = self._server.scene.add_transform_controls(
            name=self._picking_control_path(layer_id),
            scale=max(float(layer.scene_scale), 0.15),
            disable_rotations=True,
            depth_test=False,
            position=target,
        )

        @handle.on_update
        def _on_update(event, picked_layer_id=layer_id):
            self._interaction_events.put(
                (
                    "picking_target",
                    picked_layer_id,
                    tuple(float(v) for v in event.target.position),
                )
            )

        @handle.on_drag_end
        def _on_drag_end(_event, picked_layer_id=layer_id):
            self._interaction_events.put(("picking_release", picked_layer_id))

        self._picking_controls[layer_id] = handle

    def _start_picking(self, layer_id: str, ray_origin, ray_direction) -> None:
        """Raycast a click and show a Viser handle for a successful pick."""
        layer = self._layers.get(layer_id)
        if layer is None or layer.picking is None or layer._last_state is None:
            return
        self._remove_picking_control(layer_id, release=True)
        origin, direction = self._transform_ray_to_layer(layer, ray_origin, ray_direction)
        layer.picking.pick(layer._last_state, origin, direction)
        if layer.picking.is_picking():
            self._create_picking_control(layer_id)

    def _set_picking_target(self, layer_id: str, visual_target) -> None:
        """Update a picking spring target from a Viser control position."""
        layer = self._layers.get(layer_id)
        if layer is None or layer.picking is None or not layer.picking.is_picking():
            return
        layer_inv = wp.transform_inverse(layer.xform)
        point_with_offset = wp.transform_point(
            layer_inv,
            wp.vec3(float(visual_target[0]), float(visual_target[1]), float(visual_target[2])),
        )
        offset = self._picking_world_offset(layer, layer.picking)
        target = (
            np.asarray((point_with_offset[0], point_with_offset[1], point_with_offset[2]), dtype=np.float32) - offset
        )
        pick_state = layer.picking.pick_state.numpy()
        pick_state[0]["picking_target_world"] = target
        layer.picking.pick_state.assign(pick_state)

    def _process_interaction_events(self) -> None:
        """Apply browser callbacks on the simulation thread between frames."""
        while True:
            try:
                event = self._interaction_events.get_nowait()
            except queue.Empty:
                break

            event_type = event[0]
            if event_type == "gizmo_update":
                _, name, position, wxyz = event
                entry = self._gizmo_handles.get(name)
                if entry is not None:
                    self._assign_transform(entry["transform"], position, wxyz)
                    self._sync_gizmo_handles(entry)
            elif event_type == "gizmo_drag_start":
                name = event[1]
                if name in self._gizmo_handles:
                    self._active_gizmos.add(name)
                    self.gizmo_is_using = True
            elif event_type == "gizmo_drag_end":
                name = event[1]
                entry = self._gizmo_handles.get(name)
                if entry is not None and entry["snap_to"] is not None:
                    entry["transform"][:] = entry["snap_to"]
                    self._sync_gizmo_handles(entry)
                self._active_gizmos.discard(name)
                self.gizmo_is_using = bool(self._active_gizmos)
            elif event_type == "picking_click":
                _, layer_id, ray_origin, ray_direction = event
                if self.picking_enabled:
                    self._start_picking(layer_id, ray_origin, ray_direction)
            elif event_type == "picking_target":
                _, layer_id, visual_target = event
                self._set_picking_target(layer_id, visual_target)
            elif event_type == "picking_release":
                self._remove_picking_control(event[1], release=True)
            elif event_type == "viewer_option":
                _, attribute, value = event
                if attribute == "wireframe":
                    self._set_wireframe(value)
                else:
                    setattr(self, attribute, value)
            elif event_type == "step":
                self._step_requested = True
            elif event_type == "reset":
                if self._reset_callback is not None:
                    self._reset_callback()
            elif event_type == "example_switch":
                if self._example_switch_callback is not None:
                    self._example_switch_callback(event[1])

    @property
    def url(self) -> str:
        """Get the browser URL of the viser server.

        Returns:
            str: Browser URL for the running viser server.
        """
        if self.is_jupyter_notebook:
            return self._build_browser_url(self._port)
        return f"http://localhost:{self._port}"

    @staticmethod
    def _normalize_jupyter_base_url(base_url: str) -> str:
        """Normalize a Jupyter base URL for proxy path construction."""
        if not base_url:
            return ""
        if not base_url.startswith("/"):
            base_url = "/" + base_url
        if base_url != "/":
            return base_url.rstrip("/")
        return ""

    @classmethod
    def _get_jupyter_proxy_base_url(cls) -> str | None:
        """Return the Jupyter base URL when notebook proxying is available."""
        try:
            from importlib.util import find_spec  # noqa: PLC0415

            if find_spec("jupyter_server_proxy") is None:
                return None
        except Exception:
            return None

        # JUPYTER_BASE_URL is not always exported (e.g. CLI --NotebookApp.base_url).
        for env_name in ("JUPYTER_BASE_URL", "JUPYTERHUB_SERVICE_PREFIX", "NB_PREFIX"):
            candidate = os.environ.get(env_name)
            if candidate:
                return cls._normalize_jupyter_base_url(candidate)

        try:
            from jupyter_server.serverapp import list_running_servers  # noqa: PLC0415

            for server in list_running_servers():
                candidate = server.get("base_url")
                if candidate:
                    return cls._normalize_jupyter_base_url(candidate)
        except Exception:
            pass

        return None

    @classmethod
    def _build_browser_url(
        cls,
        port: int,
        *,
        path: str = "/",
        query: str = "",
        local_host: str = "localhost",
    ) -> str:
        """Build a browser URL, preferring Jupyter proxy routes when available."""
        normalized_path = "/" if path in ("", "/") else "/" + path.lstrip("/")

        jupyter_base_url = cls._get_jupyter_proxy_base_url()
        if jupyter_base_url is not None:
            return f"{jupyter_base_url}/proxy/{port}{normalized_path}{query}"

        return f"http://{local_host}:{port}{normalized_path}{query}"

    @staticmethod
    def _get_colab_output() -> Any | None:
        """Return google.colab.output when Colab iframe display is available."""
        try:
            from google.colab import output  # noqa: PLC0415
        except Exception:
            return None

        if not hasattr(output, "serve_kernel_port_as_iframe"):
            return None

        return output

    @classmethod
    def _display_colab_iframe(
        cls,
        port: int,
        *,
        path: str = "/",
        query: str = "",
        width: int | str = "100%",
        height: int | str = 400,
    ) -> bool:
        """Display a local server port in Colab when the Colab helper is available."""
        output = cls._get_colab_output()
        if output is None:
            return False

        normalized_path = "/" if path in ("", "/") else "/" + path.lstrip("/")
        normalized_query = ""
        if query:
            normalized_query = query if query.startswith("?") else "?" + query
        iframe_path = f"{normalized_path}{normalized_query}"

        try:
            output.serve_kernel_port_as_iframe(port, path=iframe_path, width=width, height=height)
        except TypeError:
            output.serve_kernel_port_as_iframe(port, path=iframe_path, height=height)

        return True

    @classmethod
    def _colab_iframe_target_from_url(cls, player_url: str) -> tuple[int, str, str] | None:
        """Extract a Colab iframe target from an absolute or Jupyter proxy URL."""
        parsed_url = urlparse(player_url)

        proxy_target = None
        path_segments = parsed_url.path.split("/")
        for index, segment in enumerate(path_segments[:-1]):
            if segment != "proxy":
                continue

            port_text = path_segments[index + 1]
            if not port_text.isdecimal():
                continue

            proxy_port = int(port_text)
            if proxy_port > 65535:
                continue

            downstream_segments = path_segments[index + 2 :]
            if not downstream_segments or downstream_segments == [""]:
                downstream_path = "/"
            else:
                downstream_path = "/" + "/".join(downstream_segments)
            proxy_target = (proxy_port, downstream_path, parsed_url.query)

        if proxy_target is not None:
            return proxy_target

        try:
            parsed_port = parsed_url.port
        except ValueError:
            parsed_port = None

        if parsed_port is not None:
            return parsed_port, parsed_url.path or "/", parsed_url.query

        return None

    @classmethod
    def get_viser_client_dir(cls) -> Path:
        """Return the installed Viser client build for notebook playback."""
        viser = cls._get_viser()

        try:
            viser_package_dir = Path(inspect.getfile(viser)).resolve().parent
        except Exception:
            viser_package_dir = None

        if viser_package_dir is not None:
            # Prefer the client build shipped with the installed viser package so
            # the playback UI matches the serializer that generated the .viser file.
            for candidate in (
                viser_package_dir / "client" / "build",
                viser_package_dir / "static",
            ):
                if (candidate / "index.html").exists():
                    return candidate.resolve()

        raise FileNotFoundError(
            "Viser client files not found in the installed viser package. "
            "The notebook playback feature requires a Viser client build."
        )

    @staticmethod
    def _is_client_camera_ready(client: Any) -> bool:
        """Return True if the client has reported an initial camera state."""
        try:
            update_timestamp = float(client.camera.update_timestamp)
        except Exception:
            # Older viser versions may not expose update_timestamp.
            try:
                _ = client.camera.position
            except Exception:
                return False
            return True
        return update_timestamp > 0.0

    def _handle_client_connect(self, client: Any):
        """Apply cached camera settings to newly connected clients."""
        self._pending_camera_clients.discard(int(client.client_id))
        self._apply_camera_to_client(client)

    def _handle_client_disconnect(self, client: Any):
        """Clear pending camera state for disconnected clients."""
        self._pending_camera_clients.discard(int(client.client_id))

    def _get_camera_up_axis(self) -> int:
        """Resolve the model up-axis as an integer index (0, 1, 2)."""
        if self.model is None:
            return 2
        up_axis = self.model.up_axis
        if isinstance(up_axis, str):
            return "XYZ".index(up_axis.upper())
        return int(up_axis)

    def _compute_camera_front_up(self, pitch: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute camera front and up vectors from pitch/yaw angles."""
        pitch = float(np.clip(pitch, -89.0, 89.0))
        yaw = float(yaw)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        up_axis = self._get_camera_up_axis()

        if up_axis == 0:  # X up
            front = np.array(
                [
                    np.sin(pitch_rad),
                    np.cos(yaw_rad) * np.cos(pitch_rad),
                    np.sin(yaw_rad) * np.cos(pitch_rad),
                ],
                dtype=np.float64,
            )
            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif up_axis == 2:  # Z up
            front = np.array(
                [
                    np.cos(yaw_rad) * np.cos(pitch_rad),
                    np.sin(yaw_rad) * np.cos(pitch_rad),
                    np.sin(pitch_rad),
                ],
                dtype=np.float64,
            )
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:  # Y up
            front = np.array(
                [
                    np.cos(yaw_rad) * np.cos(pitch_rad),
                    np.sin(pitch_rad),
                    np.sin(yaw_rad) * np.cos(pitch_rad),
                ],
                dtype=np.float64,
            )
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        front /= np.linalg.norm(front)
        right = np.cross(front, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1.0e-8:
            fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if np.linalg.norm(np.cross(front, fallback_up)) < 1.0e-8:
                fallback_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right = np.cross(front, fallback_up)
            right_norm = np.linalg.norm(right)
        right /= right_norm

        up = np.cross(right, front)
        up /= np.linalg.norm(up)
        return front, up

    def _apply_camera_to_client(self, client: Any):
        """Apply the cached camera request to a connected client if ready."""
        if self._camera_request is None:
            return

        client_id = int(client.client_id)
        if not self._is_client_camera_ready(client):
            if client_id in self._pending_camera_clients:
                return

            self._pending_camera_clients.add(client_id)

            def _on_camera_update(_camera: Any):
                if client_id not in self._pending_camera_clients:
                    return
                self._pending_camera_clients.discard(client_id)
                self._apply_camera_to_client(client)

            client.camera.on_update(_on_camera_update)
            return

        self._pending_camera_clients.discard(client_id)
        position, look_at, up_direction = self._camera_request

        # Keep camera updates synchronized to avoid transient jitter.
        if hasattr(client, "atomic"):
            with client.atomic():
                client.camera.position = tuple(position.tolist())
                client.camera.look_at = tuple(look_at.tolist())
                client.camera.up_direction = tuple(up_direction.tolist())
        else:
            client.camera.position = tuple(position.tolist())
            client.camera.look_at = tuple(look_at.tolist())
            client.camera.up_direction = tuple(up_direction.tolist())

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        """Set camera position and orientation for connected Viser clients.

        The requested view is also cached so that newly connected clients receive
        the same camera setup as soon as they report camera state.

        Args:
            pos: Requested camera position.
            pitch: Requested camera pitch angle.
            yaw: Requested camera yaw angle.
        """
        position = np.asarray((float(pos[0]), float(pos[1]), float(pos[2])), dtype=np.float64)
        front, up_direction = self._compute_camera_front_up(pitch, yaw)
        look_at = position + front
        self._camera_request = (position, look_at, up_direction)

        if hasattr(self._server, "initial_camera"):
            self._server.initial_camera.position = tuple(position.tolist())
            self._server.initial_camera.look_at = tuple(look_at.tolist())
            if hasattr(self._server.initial_camera, "up"):
                self._server.initial_camera.up = tuple(up_direction.tolist())
            elif hasattr(self._server.initial_camera, "up_direction"):
                self._server.initial_camera.up_direction = tuple(up_direction.tolist())

        for client in self._server.get_clients().values():
            self._apply_camera_to_client(client)

    @staticmethod
    def _camera_query_from_request(camera_request: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> str:
        """Build URL query parameters for playback initial camera overrides."""
        if camera_request is None:
            return ""

        position, look_at, up_direction = camera_request

        def _format_vec3(values: np.ndarray) -> str:
            values = np.asarray(values, dtype=np.float64).reshape(3)
            return ",".join(f"{float(v):.9g}" for v in values)

        return (
            f"&initialCameraPosition={_format_vec3(position)}"
            f"&initialCameraLookAt={_format_vec3(look_at)}"
            f"&initialCameraUp={_format_vec3(up_direction)}"
        )

    @override
    def log_gizmo(
        self,
        name: str,
        transform: wp.transform,
        *,
        translate: Sequence[Axis] | None = None,
        rotate: Sequence[Axis] | None = None,
        snap_to: wp.transform | None = None,
    ) -> None:
        """Log or update an interactive Viser transform gizmo.

        Translation arrows and rotation rings are separate, synchronized
        controls so each operation can honor its own enabled-axis set.

        Args:
            name: Unique gizmo path/name.
            transform: Gizmo world transform, mutated in place on interaction.
            translate: Axes on which translation handles are shown.
            rotate: Axes on which rotation rings are shown.
            snap_to: Optional world transform applied when dragging ends.
        """
        name = self._qualify(name)
        translate_axes = self._normalize_gizmo_axes(translate)
        rotate_axes = self._normalize_gizmo_axes(rotate)
        self._gizmo_seen.add(name)

        entry = self._gizmo_handles.get(name)
        if entry is not None and (entry["translate"] != translate_axes or entry["rotate"] != rotate_axes):
            self._remove_gizmo(name)
            entry = None
        if entry is None:
            entry = self._create_gizmo(name, transform, translate_axes, rotate_axes, snap_to)
        else:
            entry["transform"] = transform
            entry["snap_to"] = snap_to

        if name not in self._active_gizmos:
            self._sync_gizmo_handles(entry)

    @override
    def log_mesh(
        self,
        name: str,
        points: wp.array[wp.vec3],
        indices: wp.array[wp.int32] | wp.array[wp.uint32],
        normals: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """
        Log a mesh to viser for visualization.

        Args:
            name: Entity path for the mesh.
            points: Vertex positions.
            indices: Triangle indices.
            normals: Vertex normals, unused in viser.
            uvs: UV coordinates, used for textures if supported.
            texture: Texture path/URL or image array (H, W, C).
            hidden: Whether the mesh is hidden.
            backface_culling: Whether to enable backface culling.
            color: Optional base color as an RGB tuple with values in
                [0, 1]. Used when no texture is provided.
            roughness: Surface roughness in ``[0, 1]``. ``0`` is perfectly
                smooth, ``1`` is fully rough.
            metallic: Metallicity in ``[0, 1]``. ``0`` is dielectric, ``1``
                is metal.
        """
        name = self._qualify(name)

        assert isinstance(points, wp.array)
        assert isinstance(indices, wp.array)

        # Convert to numpy arrays
        points_np = self._to_numpy(points).astype(np.float32)
        indices_np = self._to_numpy(indices).astype(np.uint32)
        uvs_np = self._to_numpy(uvs).astype(np.float32) if uvs is not None else None
        texture_image = self._prepare_texture(texture)

        if texture_image is not None and uvs_np is None:
            warnings.warn(f"Mesh {name} has a texture but no UVs; texture will be ignored.", stacklevel=2)
            texture_image = None
        if texture_image is not None and uvs_np is not None and len(uvs_np) != len(points_np):
            warnings.warn(
                f"Mesh {name} has {len(uvs_np)} UVs for {len(points_np)} vertices; texture will be ignored.",
                stacklevel=2,
            )
            texture_image = None

        # Viser expects indices as (N, 3) for triangles
        if indices_np.ndim == 1:
            indices_np = indices_np.reshape(-1, 3)

        trimesh_mesh = None
        if texture_image is not None and uvs_np is not None:
            trimesh_mesh = self._build_trimesh_mesh(points_np, indices_np, uvs_np, texture_image)
            if trimesh_mesh is None:
                warnings.warn(
                    "Viser textured meshes require trimesh; falling back to untextured rendering.",
                    stacklevel=2,
                )

        # Store mesh data for instancing
        self._meshes[name] = {
            "points": points_np,
            "indices": indices_np,
            "uvs": uvs_np,
            "texture": texture_image,
            "trimesh": trimesh_mesh,
        }

        # Remove existing mesh if present
        if name in self._scene_handles:
            try:
                self._scene_handles[name].remove()
            except Exception:
                pass

        if hidden:
            return

        # Add mesh to viser scene
        if trimesh_mesh is not None:
            handle = self._call_scene_method(
                self._server.scene.add_mesh_trimesh,
                name=name,
                mesh=trimesh_mesh,
            )
        else:
            handle = self._call_scene_method(
                self._server.scene.add_mesh_simple,
                name=name,
                vertices=points_np,
                faces=indices_np,
                color=(180, 180, 180) if color is None else color,
                wireframe=False,
                side="double" if not backface_culling else "front",
            )
        self._scene_handles[name] = handle

    @staticmethod
    def _quats_xyzw_to_wxyz(quats_xyzw: np.ndarray) -> np.ndarray:
        """Convert quaternions from Warp's XYZW layout to viser's WXYZ layout."""
        quats_xyzw = np.asarray(quats_xyzw, dtype=np.float32)
        was_1d = quats_xyzw.ndim == 1
        if was_1d:
            quats_xyzw = quats_xyzw.reshape(1, 4)
        if quats_xyzw.ndim != 2 or quats_xyzw.shape[1] != 4:
            raise ValueError(f"Expected quaternion array with shape (4,) or (N, 4), got {quats_xyzw.shape}")

        quats_wxyz = np.empty_like(quats_xyzw)
        quats_wxyz[:, 0] = quats_xyzw[:, 3]
        quats_wxyz[:, 1] = quats_xyzw[:, 0]
        quats_wxyz[:, 2] = quats_xyzw[:, 1]
        quats_wxyz[:, 3] = quats_xyzw[:, 2]
        return quats_wxyz[0] if was_1d else quats_wxyz

    def _remove_plane_handles(self, name: str):
        """Remove any plane-grid handles associated with an instance batch."""
        handle = self._plane_handles.pop(name, None)
        if handle is None:
            return
        handles = handle if isinstance(handle, (list, tuple)) else (handle,)
        for handle in handles:
            self._detach_picking_callback(handle)
            try:
                handle.remove()
            except Exception:
                pass

    @staticmethod
    def _plane_cell_size(width: float, length: float) -> float:
        """Pick a grid cell size for a plane of the given extents."""
        spacing = max(min(float(width), float(length)) / 10.0, 0.25)
        return min(spacing, 2.0)

    def _log_plane_instances(
        self,
        name: str,
        plane_info: dict[str, float | bool],
        xforms: wp.array[wp.transform] | None,
        scales: wp.array[wp.vec3] | None,
        hidden: bool = False,
    ):
        """Render plane instances as persistent Viser grids."""
        existing = self._plane_handles.get(name)
        handles = (
            list(existing) if isinstance(existing, (list, tuple)) else ([existing] if existing is not None else [])
        )
        if hidden or xforms is None:
            for handle in handles:
                self._set_handle_property_if_changed(handle, "visible", False)
            return

        shape_batch = self._shape_instance_batch(name)
        if handles and shape_batch is not None and shape_batch.static:
            for handle in handles:
                self._set_handle_property_if_changed(handle, "visible", True)
            return

        xforms_np = self._to_numpy(xforms)
        if xforms_np is None or len(xforms_np) == 0:
            for handle in handles:
                self._set_handle_property_if_changed(handle, "visible", False)
            return

        xforms_np = np.asarray(xforms_np, dtype=np.float32)
        positions = xforms_np[:, :3]
        quats_wxyz = self._quats_xyzw_to_wxyz(xforms_np[:, 3:7])
        scales_np = self._to_numpy(scales) if scales is not None else None
        if scales_np is not None:
            scales_np = np.asarray(scales_np, dtype=np.float32)

        base_width = float(plane_info["width"])
        base_length = float(plane_info["length"])

        if len(handles) != len(positions):
            self._remove_plane_handles(name)
            handles = []

        for idx, (position, quat_wxyz) in enumerate(zip(positions, quats_wxyz, strict=False)):
            width = base_width
            length = base_length

            # TODO: Compute cell size for the scaled width/length directly
            cell_size = self._plane_cell_size(width, length)

            if scales_np is not None:
                sx = float(scales_np[idx][0])
                sy = float(scales_np[idx][1])
                width *= sx
                length *= sy
                cell_size *= max(sx, sy)

            if idx < len(handles):
                handle = handles[idx]
                self._set_handle_property_if_changed(handle, "position", position)
                self._set_handle_property_if_changed(handle, "wxyz", quat_wxyz)
                self._set_handle_property_if_changed(handle, "width", width)
                self._set_handle_property_if_changed(handle, "height", length)
                self._set_handle_property_if_changed(handle, "cell_size", cell_size)
                self._set_handle_property_if_changed(handle, "section_size", cell_size)
                self._set_handle_property_if_changed(handle, "visible", True)
            else:
                # The plane's local frame has its normal along +Z, so the grid lies in the local XY plane.
                handle = self._call_scene_method(
                    self._server.scene.add_grid,
                    name=f"{name}/grid_{idx}",
                    width=width,
                    height=length,
                    plane="xy",
                    cell_color=(150, 150, 150),
                    section_color=(110, 110, 110),
                    cell_size=cell_size,
                    section_size=cell_size,
                    position=tuple(float(v) for v in position),
                    wxyz=tuple(float(v) for v in quat_wxyz),
                )
                handles.append(handle)

        if handles:
            self._plane_handles[name] = handles

    @override
    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array[wp.transform] | None,
        scales: wp.array[wp.vec3] | None,
        colors: wp.array[wp.vec3] | None,
        materials: wp.array[wp.vec4] | None,
        hidden: bool = False,
    ):
        """
        Log instanced mesh data to viser using efficient batched rendering.

        Uses viser's add_batched_meshes_simple for GPU-accelerated instanced rendering.
        Supports in-place updates of transforms for real-time animation.

        Args:
            name: Entity path for the instances.
            mesh: Name of the mesh asset to instance.
            xforms: Instance transforms.
            scales: Instance scales.
            colors: Instance colors.
            materials: Instance materials.
            hidden: Whether the instances are hidden.
        """
        name = self._qualify(name)
        mesh = self._qualify(mesh)

        if mesh in self._plane_meshes:
            self._log_plane_instances(name, self._plane_meshes[mesh], xforms, scales, hidden=hidden)
            return

        self._remove_plane_handles(name)

        # Check that mesh exists
        if mesh not in self._meshes:
            raise RuntimeError(f"Mesh {mesh} not found. Call log_mesh first.")

        mesh_data = self._meshes[mesh]
        base_points = mesh_data["points"]
        base_indices = mesh_data["indices"]
        base_uvs = mesh_data.get("uvs")
        texture_image = self._prepare_texture(mesh_data.get("texture"))
        trimesh_mesh = mesh_data.get("trimesh")

        if hidden:
            # Remove existing instances if present
            if name in self._scene_handles:
                self._detach_picking_callback(self._scene_handles[name])
                try:
                    self._scene_handles[name].remove()
                except Exception:
                    pass
                del self._scene_handles[name]
                if name in self._instances:
                    del self._instances[name]
            return

        # Convert transforms and properties to numpy
        if xforms is None:
            return

        existing = self._instances.get(name) if name in self._scene_handles else None
        shape_batch = self._shape_instance_batch(name)
        if existing is not None and shape_batch is not None and shape_batch.static:
            positions = existing["positions"]
            quats_wxyz = existing["wxyzs"]
            num_instances = existing["count"]
        else:
            xforms_np = self._to_numpy(xforms)
            num_instances = len(xforms_np)
            positions = xforms_np[:, :3].astype(np.float32)
            quats_wxyz = self._quats_xyzw_to_wxyz(xforms_np[:, 3:7])

        if existing is not None and shape_batch is not None:
            batched_scales = existing["scales"]
        else:
            scales_np = self._to_numpy(scales) if scales is not None else None
            if scales_np is not None:
                batched_scales = scales_np.astype(np.float32)
            else:
                batched_scales = np.ones((num_instances, 3), dtype=np.float32)
        colors_np = self._to_numpy(colors) if colors is not None else None

        # Prepare colors (convert from 0-1 float to 0-255 uint8)
        if colors_np is not None:
            batched_colors = (colors_np * 255).astype(np.uint8)
        else:
            batched_colors = None  # Will use cached colors or default gray

        # Check if we already have a batched mesh handle for this name
        use_trimesh = trimesh_mesh is not None and texture_image is not None and base_uvs is not None
        if name in self._instances and name in self._scene_handles:
            # Update existing batched mesh in-place (much faster)
            handle = self._scene_handles[name]
            prev_count = self._instances[name]["count"]
            prev_use_trimesh = self._instances[name].get("use_trimesh", False)

            # If instance count changed, we need to recreate the mesh
            if prev_count != num_instances or prev_use_trimesh != use_trimesh:
                self._detach_picking_callback(handle)
                try:
                    handle.remove()
                except Exception:
                    pass
                del self._scene_handles[name]
                del self._instances[name]
            else:
                # Update transforms in-place
                try:
                    instance = self._instances[name]
                    if not np.array_equal(instance["positions"], positions):
                        handle.batched_positions = positions
                        instance["positions"] = positions.copy()
                    if not np.array_equal(instance["wxyzs"], quats_wxyz):
                        handle.batched_wxyzs = quats_wxyz
                        instance["wxyzs"] = quats_wxyz.copy()
                    if hasattr(handle, "batched_scales") and not np.array_equal(instance["scales"], batched_scales):
                        handle.batched_scales = batched_scales
                        instance["scales"] = batched_scales.copy()
                    # Only update colors if they were explicitly provided
                    if batched_colors is not None and hasattr(handle, "batched_colors"):
                        if not np.array_equal(instance["colors"], batched_colors):
                            handle.batched_colors = batched_colors
                            instance["colors"] = batched_colors.copy()
                    return
                except Exception:
                    # If update fails, recreate the mesh
                    self._detach_picking_callback(handle)
                    try:
                        handle.remove()
                    except Exception:
                        pass
                    del self._scene_handles[name]
                    del self._instances[name]

        # For new instances, use provided colors or default gray
        if batched_colors is None:
            batched_colors = np.full((num_instances, 3), 180, dtype=np.uint8)

        # Create new batched mesh.
        # LOD is disabled because viser's automatic mesh simplification creates
        # holes in complex geometry (e.g. terrain), causing popping artifacts.
        if use_trimesh:
            handle = self._call_scene_method(
                self._server.scene.add_batched_meshes_trimesh,
                name=name,
                mesh=trimesh_mesh,
                batched_positions=positions,
                batched_wxyzs=quats_wxyz,
                batched_scales=batched_scales,
                lod="off",
            )
        else:
            handle = self._call_scene_method(
                self._server.scene.add_batched_meshes_simple,
                name=name,
                vertices=base_points,
                faces=base_indices,
                batched_positions=positions,
                batched_wxyzs=quats_wxyz,
                batched_scales=batched_scales,
                batched_colors=batched_colors,
                lod="off",
                wireframe=self._wireframe,
            )

        pickable = self._is_pickable_instance(name)
        self._scene_handles[name] = handle
        self._instances[name] = {
            "mesh": mesh,
            "count": num_instances,
            "positions": positions.copy(),
            "wxyzs": quats_wxyz.copy(),
            "scales": batched_scales.copy(),
            "colors": batched_colors.copy(),
            "use_trimesh": use_trimesh,
            "layer_id": self.layer.layer_id,
            "pickable": pickable,
        }
        if pickable:
            self._attach_picking_callback(handle, self.layer.layer_id)

    @override
    def log_state(self, state: newton.State) -> None:
        """Log simulation state and update the interactive picking line.

        Args:
            state: Current simulation state.
        """
        self._last_state = state
        super().log_state(state)
        self._render_picking_line()

    def _render_picking_line(self) -> None:
        """Render the cyan line from the body anchor to the picking target."""
        if not self.picking_enabled or self.picking is None or not self.picking.is_picking():
            self.log_lines("picking_line", None, None, None)
            return
        body_idx = int(self.picking.pick_body.numpy()[0])
        if body_idx < 0:
            self.log_lines("picking_line", None, None, None)
            return

        pick_state = self.picking.pick_state.numpy()
        anchor = self._picking_point_to_visual(self.layer, self.picking, pick_state[0]["picked_point_world"])
        target = self._picking_point_to_visual(self.layer, self.picking, pick_state[0]["picking_target_world"])
        self.log_lines(
            "picking_line",
            np.asarray([anchor], dtype=np.float32),
            np.asarray([target], dtype=np.float32),
            (0.0, 1.0, 1.0),
            hidden=False,
        )

    @override
    def begin_frame(self, time: float):
        """
        Begin a new frame.

        Args:
            time: The current simulation time.
        """
        self._process_interaction_events()
        self._sync_gui_controls()
        self._gizmo_seen.clear()
        self._frame_dt = time - self.time
        self.time = time
        self._begin_atomic_frame()

    @override
    def should_step(self) -> bool:
        """Apply browser interactions before the simulation advances.

        Returns:
            bool: Whether the simulation should advance.
        """
        self._process_interaction_events()
        self._sync_gui_controls()
        if not self._paused:
            self._step_requested = False
            return True
        if self._step_requested:
            self._step_requested = False
            return True
        return False

    @override
    def is_paused(self) -> bool:
        """Return whether simulation stepping is paused."""
        return self._paused

    @override
    def end_frame(self):
        """
        End the current frame.

        If recording is active, inserts a sleep command for playback timing.
        Updates scalar plots if any data changed since last frame.
        """
        try:
            self._update_scalar_plots()

            for name in set(self._gizmo_handles) - self._gizmo_seen:
                self._remove_gizmo(name)

            if self._serializer is not None:
                # Insert sleep for frame timing during recording
                self._serializer.insert_sleep(self._frame_dt)
        finally:
            self._end_atomic_frame()

    def _update_scalar_plots(self):
        """Create or update uPlot chart handles for dirty scalar signals."""
        if not self._scalar_dirty:
            return

        try:
            from viser import uplot

            # Create the plots folder on first use.
            if self._plot_folder is None:
                self._plot_folder = self._server.gui.add_folder("Plots")

            n = self._plot_history_size
            x = np.arange(n, dtype=np.float64)

            for name in self._scalar_dirty:
                buf = self._scalar_buffers[name]
                y = np.full(n, np.nan, dtype=np.float64)
                y[n - len(buf) :] = np.array(buf, dtype=np.float64)

                handle = self._plot_handles.get(name)
                if handle is None:
                    with self._plot_folder:
                        handle = self._server.gui.add_uplot(
                            data=(x, y),
                            series=(
                                uplot.Series(label="step"),
                                uplot.Series(label=name, stroke="#3b82f6", width=2),
                            ),
                            scales={"x": uplot.Scale(time=False)},
                            aspect=2.0,
                        )
                    self._plot_handles[name] = handle
                else:
                    handle.data = (x, y)
        except Exception:
            pass

        self._scalar_dirty.clear()

    @override
    def is_running(self) -> bool:
        """
        Check if the viewer is still running.

        Returns:
            bool: True if the viewer is running, False otherwise.
        """
        return self._running

    @override
    def close(self):
        """
        Close the viewer and clean up resources.
        """
        self._running = False
        self._end_atomic_frame()
        for name in list(self._gizmo_handles):
            self._remove_gizmo(name)
        for layer_id in list(self._picking_controls):
            self._remove_picking_control(layer_id, release=True)
        try:
            self._server.stop()
            if self._serializer is not None:
                self.save_recording()
        except Exception:
            pass

    @override
    def apply_forces(self, state: newton.State):
        """Apply the force from an active Viser picking handle.

        Args:
            state: Current simulation state.
        """
        if self.picking_enabled and self.picking is not None:
            self.picking._apply_picking_force(state)

    def save_recording(self):
        """
        Save the current recording to a .viser file.

        The recording can be played back in a static HTML viewer.
        See build_static_viewer() for creating the HTML player.

        Note:
            Recording must be enabled by passing ``record_to_viser`` to the constructor.

        Example:

            .. code-block:: python

                viewer = ViewerViser(record_to_viser="my_simulation.viser")
                # ... run simulation ...
                viewer.save_recording()
        """
        if self._serializer is None or self._record_to_viser is None:
            raise RuntimeError("No recording in progress. Pass record_to_viser to the constructor.")

        from pathlib import Path  # noqa: PLC0415

        data = self._serializer.serialize()
        Path(self._record_to_viser).write_bytes(data)

        self._serializer = None

        if self.verbose:
            print(f"Recording saved to: {self._record_to_viser}")

    @override
    def log_lines(
        self,
        name: str,
        starts: wp.array[wp.vec3] | None,
        ends: wp.array[wp.vec3] | None,
        colors: (wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None),
        width: float = 0.01,
        hidden: bool = False,
    ):
        """
        Log lines for visualization.

        Args:
            name: Name of the line batch.
            starts: Line start points.
            ends: Line end points.
            colors: Line colors.
            width: Line width.
            hidden: Whether the lines are hidden.
        """
        name = self._qualify(name)

        def remove_existing_line(reset_version: bool = True):
            handle = self._scene_handles.pop(name, None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass
            self._line_segment_counts.pop(name, None)
            if reset_version:
                self._line_versions.pop(name, None)

        if hidden:
            remove_existing_line()
            return

        if starts is None or ends is None:
            remove_existing_line()
            return

        starts_np = self._to_numpy(starts)
        ends_np = self._to_numpy(ends)

        if starts_np is None or ends_np is None or len(starts_np) == 0:
            remove_existing_line()
            return

        starts_np = np.asarray(starts_np, dtype=np.float32)
        ends_np = np.asarray(ends_np, dtype=np.float32)
        num_lines = len(starts_np)

        # Viser requires points with shape (N, 2, 3): [start, end] per segment.
        line_points = np.stack((starts_np, ends_np), axis=1)

        def _rgb_to_uint8_array(rgb: np.ndarray) -> np.ndarray:
            rgb = np.asarray(rgb, dtype=np.float32)
            max_val = float(np.max(rgb)) if rgb.size > 0 else 0.0
            if max_val <= 1.0:
                rgb = rgb * 255.0
            return np.clip(rgb, 0, 255).astype(np.uint8)

        # Process colors. Keep this as a NumPy array because Viser handle
        # property updates require the normalized array form, even though
        # add_line_segments() also accepts RGB tuples on initial creation.
        color_rgb: np.ndarray = np.array((0, 255, 0), dtype=np.uint8)
        if colors is not None:
            colors_np = self._to_numpy(colors)
            if colors_np is not None:
                colors_np = np.asarray(colors_np)
                if colors_np.ndim == 1 and colors_np.shape[0] == 3:
                    # Single color for all lines.
                    color_rgb = _rgb_to_uint8_array(colors_np)
                elif colors_np.ndim == 2 and colors_np.shape == (num_lines, 3):
                    # Per-line colors: repeat each line color for [start, end].
                    line_colors = _rgb_to_uint8_array(colors_np)
                    color_rgb = np.repeat(line_colors[:, None, :], 2, axis=1)
                elif colors_np.ndim == 3 and colors_np.shape == (num_lines, 2, 3):
                    # Already per-point-per-segment colors.
                    color_rgb = _rgb_to_uint8_array(colors_np)

        line_width = width * 100  # Scale for visibility
        if name in self._scene_handles:
            handle = self._scene_handles[name]
            if self._line_segment_counts.get(name) == num_lines:
                try:
                    handle.points = line_points
                    handle.colors = color_rgb
                    handle.line_width = line_width
                    return
                except Exception:
                    remove_existing_line()
            else:
                self._line_versions[name] = self._line_versions.get(name, 0) + 1
                remove_existing_line(reset_version=False)

        version = self._line_versions.get(name, 0)
        scene_name = name if version == 0 else f"{name}__line_{version}"

        handle = self._server.scene.add_line_segments(
            name=scene_name,
            points=line_points,
            colors=color_rgb,
            line_width=line_width,
        )
        self._scene_handles[name] = handle
        self._line_segment_counts[name] = num_lines

    @override
    def log_geo(
        self,
        name: str,
        geo_type: int,
        geo_scale: tuple[float, ...],
        geo_thickness: float,
        geo_is_solid: bool,
        geo_src: newton.Mesh | newton.Heightfield | None = None,
        hidden: bool = False,
    ):
        """Log a geometry primitive, with plane expansion for infinite planes.

        Args:
            name: Unique path/name for the geometry asset.
            geo_type: Geometry type value from `newton.GeoType`.
            geo_scale: Geometry scale tuple interpreted by `geo_type`.
            geo_thickness: Shell thickness for mesh-like geometry.
            geo_is_solid: Whether mesh geometry is treated as solid.
            geo_src: Optional source geometry for mesh-backed types.
            hidden: Whether the resulting geometry is hidden.
        """
        name = self._qualify(name)

        if geo_type == newton.GeoType.PLANE:
            # Handle "infinite" planes encoded with non-positive scales
            if geo_scale[0] == 0.0 or geo_scale[1] == 0.0:
                extents = self._get_world_extents()
                if extents is None:
                    width, length = 10.0, 10.0
                else:
                    max_extent = max(max(extents) * 1.5, 8.0)
                    width = max_extent
                    length = max_extent
            else:
                width = geo_scale[0]
                length = geo_scale[1] if len(geo_scale) > 1 else 10.0
            self._plane_meshes[name] = {
                "width": float(width),
                "length": float(length),
            }
        else:
            super().log_geo(name, geo_type, geo_scale, geo_thickness, geo_is_solid, geo_src, hidden)

    @override
    def log_points(
        self,
        name: str,
        points: wp.array[wp.vec3] | None,
        radii: wp.array[wp.float32] | float | None = None,
        colors: (wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None) = None,
        hidden: bool = False,
    ):
        """
        Log points for visualization.

        Args:
            name: Name of the point batch.
            points: Point positions (can be a wp.array or a numpy array).
            radii: Point radii (can be a wp.array or a numpy array).
            colors: Point colors (can be a wp.array or a numpy array).
            hidden: Whether the points are hidden.
        """
        name = self._qualify(name)

        # Remove existing points if present
        if name in self._scene_handles:
            try:
                self._scene_handles[name].remove()
            except Exception:
                pass

        if hidden:
            return

        if points is None:
            return

        pts = self._to_numpy(points)
        n_points = pts.shape[0]

        if n_points == 0:
            return

        # Handle radii (point size)
        if radii is not None:
            size = self._to_numpy(radii)
            if size.ndim == 0 or size.shape == ():
                point_size = float(size)
            elif len(size) == n_points:
                point_size = float(np.mean(size))  # Use average for uniform size
            else:
                point_size = 0.1
        else:
            point_size = 0.1

        # Handle colors
        if colors is not None:
            cols = self._to_numpy(colors)
            if cols.shape == (n_points, 3):
                # Convert from 0-1 to 0-255
                colors_val = (cols * 255).astype(np.uint8)
            elif cols.shape == (3,):
                colors_val = np.tile((cols * 255).astype(np.uint8), (n_points, 1))
            else:
                colors_val = np.full((n_points, 3), 255, dtype=np.uint8)
        else:
            colors_val = np.full((n_points, 3), 255, dtype=np.uint8)

        # Add point cloud to viser
        handle = self._server.scene.add_point_cloud(
            name=name,
            points=pts.astype(np.float32),
            colors=colors_val,
            point_size=point_size,
            point_shape="circle",
        )
        self._scene_handles[name] = handle

    @override
    def log_array(self, name: str, array: wp.array[Any] | np.ndarray):
        """Viser viewer does not visualize generic arrays.

        Args:
            name: Unique path/name for the array signal.
            array: Array data to visualize.
        """
        pass

    @override
    def log_scalar(
        self,
        name: str,
        value: int | float | bool | np.number,
        *,
        clear: bool = False,
        smoothing: int = 1,
    ):
        """
        Log a scalar value as a live time-series plot.

        Each unique *name* creates a separate uPlot chart in a "Plots"
        folder in the GUI sidebar.  Values are stored in a rolling
        buffer of the last ``plot_history_size`` samples.

        Args:
            name: Unique path/name for the scalar signal.
            value: Scalar value to record.
            clear: If ``True``, discard previously recorded samples for
                *name* before logging the new value.
            smoothing: Number of raw samples to average before committing
                a point to the plot history.  Defaults to ``1`` (no smoothing).
        """
        if smoothing < 1:
            raise ValueError("smoothing must be >= 1")
        name = self._qualify(name)
        val = float(value.item() if hasattr(value, "item") else value)
        buf = self._scalar_buffers.get(name)
        if buf is None:
            buf = collections.deque(maxlen=self._plot_history_size)
            self._scalar_buffers[name] = buf
        elif clear:
            buf.clear()
            self._scalar_accumulators.pop(name, None)

        if self._scalar_smoothing.get(name, smoothing) != smoothing:
            self._scalar_accumulators.pop(name, None)
        self._scalar_smoothing[name] = smoothing
        if smoothing <= 1:
            buf.append(val)
        else:
            acc = self._scalar_accumulators.get(name)
            if acc is None:
                acc = []
                self._scalar_accumulators[name] = acc
            acc.append(val)
            if len(acc) >= smoothing:
                buf.append(sum(acc) / len(acc))
                acc.clear()

        self._scalar_dirty.add(name)

    def show_notebook(self, width: int | str = "100%", height: int | str = 400):
        """
        Show the viewer in a Jupyter notebook.

        If recording is active, saves the recording and displays using the static HTML
        viewer with timeline controls. Otherwise, displays the live server in an IFrame.

        Args:
            width: Width of the embedded player in pixels.
            height: Height of the embedded player in pixels.

        Returns:
            The display object.

        Example:

            .. code-block:: python

                viewer = newton.viewer.ViewerViser(record_to_viser="my_sim.viser")
                viewer.set_model(model)
                # ... run simulation ...
                viewer.show_notebook()  # Saves recording and displays with timeline
        """

        from IPython.display import HTML, IFrame, display

        from .viewer import is_sphinx_build  # noqa: PLC0415

        if self._record_to_viser is None:
            if self._display_colab_iframe(self._port, width=width, height=height):
                return None

            # No recording - display the live server via IFrame
            return display(IFrame(src=self.url, width=width, height=height))

        if self._serializer is not None:
            # Recording is active - save it first
            recording_path = Path(self._record_to_viser)
            recording_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_recording()

        # Check if recording path contains _static - indicates Sphinx docs build
        recording_str = str(self._record_to_viser).replace("\\", "/")

        if is_sphinx_build():
            # Sphinx build - use static HTML with viser player
            # The recording path needs to be relative to the viser index.html location
            # which is at _static/viser/index.html

            # Find the _static portion of the path
            static_idx = recording_str.find("_static/")
            if static_idx == -1:
                raise ValueError(
                    f"Recordings that are supposed to appear in the Sphinx documentation must be stored in docs/_static/, but the path {recording_str} does not contain _static/"
                )
            else:
                # Extract path from _static onwards (e.g., "_static/recordings/foo.viser")
                static_relative = recording_str[static_idx:]
                # The viser index.html is at _static/viser/index.html
                # So from there, we need "../recordings/foo.viser"
                # Remove the "_static/" prefix and prepend "../"
                playback_path = "../" + static_relative[len("_static/") :]

            camera_query = self._camera_query_from_request(self._camera_request)

            embed_html = f"""
<div class="viser-player-container" style="margin: 20px 0;">
<iframe
    src="../_static/viser/index.html?playbackPath={playback_path}{camera_query}"
    width="{width}"
    height="{height}"
    frameborder="0"
    style="border: 1px solid #ccc; border-radius: 8px;">
</iframe>
</div>
"""
            return display(HTML(embed_html))
        else:
            # Regular Jupyter - use local HTTP server with viser client
            player_url = self._serve_viser_recording(
                self._record_to_viser,
                camera_request=self._camera_request,
            )
            colab_target = self._colab_iframe_target_from_url(player_url)
            if colab_target is not None:
                colab_port, colab_path, colab_query = colab_target
                if self._display_colab_iframe(
                    colab_port,
                    path=colab_path,
                    query=colab_query,
                    width=width,
                    height=height,
                ):
                    return None

            return display(IFrame(src=player_url, width=width, height=height))

    def _ipython_display_(self):
        """
        Display the viewer in an IPython notebook when the viewer is at the end of a cell.
        """
        self.show_notebook()

    @staticmethod
    def _serve_viser_recording(
        recording_path: str, camera_request: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> str:
        """
        Hosts a simple HTTP server to serve the viser recording file with the viser client
        and returns the URL of the player.

        Args:
            recording_path: Path to the .viser recording file.
            camera_request: Optional `(position, look_at, up_direction)` triple used
                to append initial camera URL overrides for playback.

        Returns:
            URL of the player.
        """
        import threading  # noqa: PLC0415
        from http.server import HTTPServer, SimpleHTTPRequestHandler  # noqa: PLC0415

        recording_path = Path(recording_path).resolve()
        if not recording_path.exists():
            raise FileNotFoundError(f"Recording file not found: {recording_path}")

        viser_client_dir = ViewerViser.get_viser_client_dir()

        # Read the recording file content
        recording_bytes = recording_path.read_bytes()

        # Create a custom HTTP handler factory that serves both viser client and the recording
        def make_handler(recording_data: bytes, client_dir: str):
            class RecordingHandler(SimpleHTTPRequestHandler):
                # Fix MIME types for JavaScript and other files (Windows often has wrong mappings)
                extensions_map: ClassVar = {  # pyright: ignore[reportIncompatibleVariableOverride]
                    **SimpleHTTPRequestHandler.extensions_map,
                    ".html": "text/html",
                    ".htm": "text/html",
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".json": "application/json",
                    ".wasm": "application/wasm",
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".ico": "image/x-icon",
                    ".ttf": "font/ttf",
                    ".hdr": "application/octet-stream",
                    ".viser": "application/octet-stream",
                    "": "application/octet-stream",
                }

                def __init__(self, *args, **kwargs):
                    self.recording_data = recording_data
                    super().__init__(*args, directory=client_dir, **kwargs)

                def do_GET(self):
                    # Parse path without query string
                    path = self.path.split("?")[0]

                    # Serve the recording file at /recording.viser
                    if path == "/recording.viser":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(len(self.recording_data)))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(self.recording_data)
                    else:
                        # Serve viser client files
                        super().do_GET()

                def log_message(self, format, *args):
                    pass  # Suppress log messages

            return RecordingHandler

        handler_class = make_handler(recording_bytes, str(viser_client_dir))
        server = HTTPServer(("127.0.0.1", 0), handler_class)
        port = server.server_address[1]

        # Start server in background thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        # Keep playbackPath relative so notebook proxy prefixes (e.g. /lab/proxy/<port>/)
        # are preserved. Each viewer instance uses a different port, so paths stay distinct.
        playback_path = "recording.viser"
        player_url = ViewerViser._build_browser_url(
            port,
            query=f"?playbackPath={playback_path}",
            local_host="127.0.0.1",
        )

        return player_url + ViewerViser._camera_query_from_request(camera_request)
