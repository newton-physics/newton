# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent viewport controls for the local simulator workspace."""

import numpy as np

from ..geometry.flags import ShapeFlags
from ..geometry.types import GeoType


class ViewportControls:
    modes = (
        ("visual", "sphere", "Visual"),
        ("collision", "collision", "Collision"),
        ("xray", "xray", "X-ray"),
        ("wire", "mesh", "Wireframe"),
    )

    def __init__(self, workspace):
        from .viewer_orientation import OrientationGizmo  # noqa: PLC0415

        self.workspace = workspace
        self.orientation = OrientationGizmo(self)
        self.mode = "visual"
        self.shell_opacity = 0.65
        self.shell_edges = True
        self.deformable_opacity = 0.28
        self.deformable_edges = True
        self.collision_opacity = 0.196
        self.collision_edges = True
        self.proximity_fade = True
        self.proximity_range = 3.0
        self.adaptive_detail = True
        self.margin_mode = 0
        self.surface_grid = True
        self._model = None
        self._inspection_model = None
        self._roles_key = None
        self._roles = ()

    def set_mode(self, mode):
        self.mode = mode
        viewer = self.workspace.viewer
        viewer.show_visual = mode in ("visual", "xray", "wire")
        viewer.show_collision = mode in ("collision", "xray")
        viewer.renderer.draw_wireframe = mode == "wire"
        viewer.renderer._physics_xray = mode == "xray"

    def sync(self):
        viewer = self.workspace.viewer
        if viewer.model is not self._model:
            self._model = viewer.model
            self.orientation._animation = None
            self.set_mode(self.mode)
            viewer.sdf_margin_mode = viewer.SDFMarginMode(self.margin_mode)
            self._inspection_model = None
            viewer.renderer._xray_inspection_radius = 0.0
        if self.mode == "xray" and self._model is not self._inspection_model:
            # One scale per model, independent of camera dolly and world count.
            # Small links and fasteners should fade alongside the larger bodies
            # they belong to. Ignore static terrain when movable shapes exist.
            radius = 0.0
            if self._model is not None and self._model.shape_count:
                radii = self._model.shape_collision_radius.numpy()
                bodies = self._model.shape_body.numpy()
                types = self._model.shape_type.numpy()
                valid = np.isfinite(radii) & (radii > 0) & (types != int(GeoType.PLANE))
                movable = valid & (bodies >= 0)
                chosen = radii[movable if np.any(movable) else valid]
                radius = float(chosen.max()) if chosen.size else 0.0
            viewer.renderer._xray_inspection_radius = radius
            self._inspection_model = self._model
        renderer = viewer.renderer
        if not hasattr(renderer, "_xray_inspection_radius"):
            renderer._xray_inspection_radius = 0.0
        renderer._physics_xray = self.mode == "xray"
        renderer._surface_grid_enabled = self.surface_grid and self.mode in ("wire", "xray")
        renderer._xray_opacity = self.shell_opacity
        renderer._xray_edges = self.shell_edges
        renderer._xray_deformable_opacity = self.deformable_opacity
        renderer._xray_deformable_edges = self.deformable_edges
        renderer._xray_collision_opacity = self.collision_opacity
        renderer._xray_collision_edges = self.collision_edges
        renderer._xray_proximity_range = self.proximity_range if self.proximity_fade else 0.0
        renderer._xray_adaptive = self.adaptive_detail
        # Shape batches already separate flags, static status and opacity.
        # Tag their existing GL objects; no geometry duplication or scene copies.
        key = (id(viewer._shape_instances), len(viewer._shape_instances), len(viewer._sdf_isomesh_instances))
        if key != self._roles_key:
            roles = []
            collide = int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)
            for batch in viewer._shape_instances.values():
                role = "collision" if batch.flags & collide else "visual"
                sdf_indices = viewer._shape_sdf_index_host
                if (
                    not batch.flags & int(ShapeFlags.COLLIDE_SHAPES)
                    and batch.flags & int(ShapeFlags.VISIBLE)
                    and sdf_indices is not None
                    and batch.model_shapes
                    and sdf_indices[batch.model_shapes[0]] >= 0
                    and viewer.model._texture_sdf_data is not None
                ):
                    role = "visual"
                if batch.geo_type == GeoType.PLANE:
                    role = "ground"
                grid = 2 if batch.geo_type == GeoType.PLANE else 1 if batch.geo_type == GeoType.BOX else 0
                roles.extend(
                    (
                        (batch.name, role, batch.static, grid),
                        (batch.name + "/__transparent__", role, batch.static, grid),
                    )
                )
            roles.extend((batch.name, "collision", batch.static, 0) for batch in viewer._sdf_isomesh_instances.values())
            self._roles, self._roles_key = roles, key
        for name, role, static, grid in self._roles:
            obj = viewer.objects.get(name)
            if obj is not None:
                obj._physics_role = role
                obj._physics_static = static
                obj._surface_grid = grid

    def camera_view(self, direction):
        camera = self.workspace.viewer.camera
        direction = np.asarray(direction, dtype=float)
        direction /= max(np.linalg.norm(direction), 1e-12)
        # A tiny tilt keeps the existing orbit camera well-defined at the poles.
        if abs(direction[camera.up_axis]) > 0.99:
            direction[(camera.up_axis + 1) % 3] = 0.018
        direction /= np.linalg.norm(direction)
        pivot, distance = camera.pivot, camera.pivot_distance
        camera.pos = type(camera.pos)(*(np.asarray(pivot) + direction * distance))
        camera.look_at(pivot)

    def draw(self):
        workspace = self.workspace
        self.orientation.draw()
        ui, widgets, viewer = workspace.gui.ui.imgui, workspace.widgets, workspace.viewer
        s = widgets.scale
        x, y, width, _ = workspace.rect
        compact = width < 275 * s
        label = "" if compact else "Guides"
        toolbar_width = 199 * s + (ui.calc_text_size(label).x + 5 * s if label else 0)
        if toolbar_width > width - 16 * s:
            return
        ui.set_next_window_pos(ui.ImVec2(x + width - toolbar_width - 8 * s, y + 8 * s))
        ui.set_next_window_size(ui.ImVec2(toolbar_width, 34 * s))
        ui.push_style_var(ui.StyleVar_.window_rounding, 7 * s)
        ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(6 * s, 5 * s))
        flags = (
            ui.WindowFlags_.no_decoration
            | ui.WindowFlags_.no_move
            | ui.WindowFlags_.no_saved_settings
            | ui.WindowFlags_.no_scroll_with_mouse
            | ui.WindowFlags_.no_background
        )
        opened = ui.begin("##viewport_controls", flags=flags)[0]
        if opened:
            pos = ui.get_window_pos()
            end = ui.ImVec2(pos.x + toolbar_width, pos.y + 34 * s)
            hovered = ui.is_mouse_hovering_rect(pos, end, False)
            expanded = ui.is_popup_open("##guides") or ui.is_popup_open("##render_settings")
            draw = ui.get_window_draw_list()
            draw.add_rect_filled(
                pos, end, widgets.color((0.12, 0.12, 0.12, 0.94 if hovered or expanded else 0.65)), 7 * s
            )
            draw.add_rect(pos, end, widgets.color((0.42, 0.42, 0.42, 0.55 if hovered or expanded else 0.28)), 7 * s)
            row_y = ui.get_cursor_pos_y()
            if widgets.icon_button(
                "##view_guides",
                "grid",
                "Choose joint and scene overlays",
                size=24,
                label=label,
                flat=not (hovered or expanded),
            ):
                ui.open_popup("##guides")
            for mode, icon, label in self.modes:
                ui.same_line(spacing=3 * s)
                ui.set_cursor_pos_y(row_y)
                if widgets.icon_button(
                    "##mode_" + mode, icon, label, active=self.mode == mode, size=24, flat=not (hovered or expanded)
                ):
                    self.set_mode(mode)
            ui.same_line(spacing=5 * s)
            ui.set_cursor_pos_y(row_y)
            if widgets.icon_button(
                "##view_contacts",
                "contact",
                "Contact points, normals and forces",
                active=viewer.show_contacts,
                size=24,
                flat=not (hovered or expanded),
            ):
                viewer.show_contacts = not viewer.show_contacts
            ui.same_line(spacing=2 * s)
            ui.set_cursor_pos_y(row_y)
            if widgets.icon_button(
                "##view_settings", "sliders", "Viewer settings", size=24, flat=not (hovered or expanded)
            ):
                ui.open_popup("##render_settings")
        ui.pop_style_var(2)
        if opened:
            for name, draw in (
                ("##guides", self._guides_menu),
                ("##render_settings", self._render_menu),
            ):
                # Auto-height, fixed width: fill-width fields must not feed back
                # into ImGui's auto-fit width on subsequent popup frames.
                self.prepare_popup(width=280)
                if ui.begin_popup(name):
                    draw()
                    ui.end_popup()
                ui.pop_style_var()
        ui.end()

    def prepare_popup(self, *, width=360):
        ui, s = self.workspace.gui.ui.imgui, self.workspace.widgets.scale
        width = min(width * s, ui.get_io().display_size.x - 24 * s)
        ui.set_next_window_size_constraints(ui.ImVec2(width, 0), ui.ImVec2(width, ui.get_io().display_size.y - 24 * s))
        ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(10 * s, 10 * s))

    def _camera_menu(self):
        workspace = self.workspace
        ui, widgets, camera = workspace.gui.ui.imgui, workspace.widgets, workspace.viewer.camera
        if widgets.button("Frame selection" if workspace.selected else "Frame scene", ui.ImVec2(-1, 0)):
            workspace.gui.frame_camera_on_model()
        if workspace.selected and widgets.button("Frame all", ui.ImVec2(-1, 0)):
            workspace.gui.frame_camera_on_model(selection=False)
        ui.separator_text("View direction")
        for axis, label in enumerate("XYZ"):
            for sign in (1, -1):
                if sign < 0:
                    ui.same_line()
                if widgets.button(("+" if sign > 0 else "-") + label, ui.ImVec2(135 * widgets.scale, 0)):
                    self.orientation.snap(np.eye(3)[axis] * sign)
        self._camera_settings()
        edit = widgets.vec3("Position", camera.pos, editable=True, unit="m")
        if edit.changed:
            camera.translate(type(camera.pos)(*edit.value) - camera.pos)

    def _guides_menu(self):
        workspace = self.workspace
        ui, viewer = workspace.gui.ui.imgui, workspace.viewer
        for title, properties in (
            (
                "Bodies",
                (
                    ("show_joints", "Joint frames"),
                    ("show_com", "Centers of mass"),
                    ("show_inertia_boxes", "Inertia boxes"),
                ),
            ),
            ("Scene", (("show_particles", "Particles"), ("show_ground", "Ground"))),
        ):
            ui.separator_text(title)
            for name, label in properties:
                changed, value = ui.checkbox(label, bool(getattr(viewer, name)))
                setattr(viewer, name, value)
                if changed and name == "show_particles":
                    viewer._particle_visibility_user = value

        ui.begin_disabled(self.mode not in ("wire", "xray"))
        _, self.surface_grid = ui.checkbox("Adaptive surface grid", self.surface_grid)
        ui.end_disabled()
        if ui.is_item_hovered(ui.HoveredFlags_.allow_when_disabled):
            ui.set_tooltip(
                "Metric grid on planes and broad box faces in X-ray and wireframe. Finer lines fade in as you approach."
            )

        ui.separator_text("Collision offsets")
        edit = workspace.widgets.choice(
            "Envelope",
            int(viewer.sdf_margin_mode),
            {0: "Off", 1: "Margin", 2: "Margin + gap"},
            editable=True,
        )
        if edit.changed:
            self.margin_mode = edit.value
            viewer.sdf_margin_mode = viewer.SDFMarginMode(edit.value)

    def _render_menu(self):
        workspace = self.workspace
        ui = workspace.gui.ui.imgui
        if ui.begin_tab_bar("##viewer_settings"):
            for label, draw in (
                ("Rendering", self._rendering_settings),
                ("Contacts", self._contact_settings),
                ("Camera", self._camera_settings),
            ):
                if workspace.widgets.tab_item(label)[0]:
                    draw()
                    ui.end_tab_item()
            ui.end_tab_bar()

    def _contact_settings(self):
        workspace = self.workspace
        ui, viewer, widgets = workspace.gui.ui.imgui, workspace.viewer, workspace.widgets
        renderer = viewer.renderer

        def number(label, owner, name, *, bounds, speed=0.01, unit=""):
            edit = widgets.number(label, getattr(owner, name), editable=True, speed=speed, bounds=bounds, unit=unit)
            if edit.changed:
                setattr(owner, name, edit.value)

        _, viewer.show_contacts = ui.checkbox("Show contacts", viewer.show_contacts)
        ui.begin_disabled(not viewer.show_contacts)
        ui.separator_text("Points")
        _, viewer.show_contact_disks = ui.checkbox("Mode disks", viewer.show_contact_disks)
        if ui.is_item_hovered():
            ui.set_tooltip(
                "Filled green: loaded, sticking. Filled blue: loaded, sliding.\n"
                "Hollow gray: unloaded. Dashed amber: force data unavailable.\n"
                "Modes use measured force and tangential speed with the thresholds below."
            )
        ui.begin_disabled(not viewer.show_contact_disks)
        number("Disk size", renderer, "contact_disk_scale", bounds=(0.1, 3.0), unit="x")
        ui.end_disabled()
        _, viewer.show_contact_pairs = ui.checkbox("Opposing points", viewer.show_contact_pairs)
        ui.begin_disabled(not viewer.show_contact_pairs)
        _, viewer.show_contact_lines = ui.checkbox("Connecting lines", viewer.show_contact_lines)
        ui.end_disabled()
        _, viewer.show_contact_labels = ui.checkbox("Contact labels", viewer.show_contact_labels)
        if ui.is_item_hovered():
            ui.set_tooltip("Numbers for the selected contacts or contact list while paused.")

        ui.separator_text("Arrows")
        _, viewer.show_contact_normals = ui.checkbox("Normal arrows", viewer.show_contact_normals)
        ui.begin_disabled(not viewer.show_contact_normals)
        number("Normal length", viewer, "contact_normal_scale", bounds=(0.05, 5.0), unit="x")
        ui.end_disabled()
        _, viewer.show_contact_forces = ui.checkbox("Force arrows", viewer.show_contact_forces)
        ui.begin_disabled(not viewer.show_contact_forces)
        gain = viewer.contact_force_scale / max(viewer._contact_force_scale_default, 1e-12)
        edit = widgets.number("Force gain", gain, editable=True, speed=0.01, bounds=(0, 20), unit="x")
        if edit.changed:
            viewer.contact_force_scale = edit.value * viewer._contact_force_scale_default
        ui.end_disabled()
        number("Arrow width", renderer, "contact_arrow_scale", bounds=(0.25, 4.0), unit="x")
        with widgets.group("Advanced", default_open=False) as opened:
            if opened:
                number("Overall scale", viewer, "contact_viz_scale", bounds=(0.01, 10), unit="x")
                number("Normal limit", renderer, "contact_normal_max_pixels", bounds=(8, 300), speed=0.5, unit="px")
                number("Force limit", renderer, "contact_force_max_pixels", bounds=(8, 300), speed=0.5, unit="px")
                ui.separator_text("Mode thresholds")
                number("Force", viewer, "contact_mode_eps_force", bounds=(0, 1e3), speed=0.0001, unit="N")
                number("Speed", viewer, "contact_mode_eps_velocity", bounds=(0, 10), speed=0.0001, unit="m/s")
        ui.end_disabled()

    def _camera_settings(self):
        workspace = self.workspace
        ui, viewer, widgets = workspace.gui.ui.imgui, workspace.viewer, workspace.widgets
        camera = viewer.camera
        ui.separator_text("Movement")
        edit = widgets.number(
            "Move speed", viewer.camera_speed, editable=True, speed=0.02, bounds=(0, 1000), unit="m/s"
        )
        if edit.changed:
            viewer.camera_speed = edit.value
        edit = widgets.number(
            "Near slowdown",
            viewer.camera_slowdown,
            editable=True,
            speed=0.02,
            bounds=(0, 8),
            reason="Higher values slow movement more near selected surfaces. 0 disables slowdown; 1 uses linear falloff.",
        )
        if edit.changed:
            viewer.camera_slowdown = edit.value
        ui.separator_text("Projection")
        edit = widgets.number("Field of view", camera.fov, editable=True, unit="°", speed=0.1, bounds=(10, 150))
        if edit.changed:
            camera.fov = edit.value
        edit = widgets.number(
            "Near clip",
            camera.near,
            editable=True,
            unit="m",
            speed=max(1e-6, camera.near * 0.01),
            bounds=(1e-5, camera.far / 1.01),
        )
        if edit.changed:
            camera.near = edit.value
        edit = widgets.number(
            "Far clip",
            camera.far,
            editable=True,
            unit="m",
            speed=max(0.001, camera.far * 0.01),
            bounds=(camera.near * 1.01, max(1e7, camera.near * 2)),
        )
        if edit.changed:
            camera.far = edit.value

    def _rendering_settings(self):
        workspace = self.workspace
        ui, viewer = workspace.gui.ui.imgui, workspace.viewer
        ui.separator_text("X-ray · Collision")
        _, self.proximity_fade = ui.checkbox("Fade nearby colliders", self.proximity_fade)
        if ui.is_item_hovered():
            ui.set_tooltip(
                "Nearby collision surfaces become transparent to expose contacts. Distant shapes stay solid."
            )
        for label, name, speed, bounds in (
            ("Near opacity", "collision_opacity", 0.002, (0.02, 0.65)),
            ("Fade distance", "proximity_range", 0.01, (0.25, 3.0)),
        ):
            edit = workspace.widgets.number(label, getattr(self, name), editable=True, speed=speed, bounds=bounds)
            if edit.changed:
                setattr(self, name, edit.value)
        _, self.collision_edges = ui.checkbox("Collision wire overlay", self.collision_edges)
        ui.separator_text("X-ray · Visual")
        edit = workspace.widgets.number(
            "Shell opacity", self.shell_opacity, editable=True, speed=0.002, bounds=(0.02, 0.65)
        )
        if edit.changed:
            self.shell_opacity = edit.value
        _, self.shell_edges = ui.checkbox("Visual wire overlay", self.shell_edges)
        _, self.adaptive_detail = ui.checkbox("Fade distant and dense detail", self.adaptive_detail)
        if ui.is_item_hovered():
            ui.set_tooltip("Reduces shell opacity with distance and fades edges as triangles become too small to read.")
        if viewer.model is not None and viewer.model.tri_count:
            ui.separator_text("X-ray · Deformables")
            edit = workspace.widgets.number(
                "Surface opacity", self.deformable_opacity, editable=True, speed=0.002, bounds=(0.02, 0.8)
            )
            if edit.changed:
                self.deformable_opacity = edit.value
            _, self.deformable_edges = ui.checkbox("Mesh edges", self.deformable_edges)
            if ui.is_item_hovered():
                ui.set_tooltip("Shows boundary triangles; select tetrahedra to inspect interior cells.")
        ui.separator_text("Lighting")
        for name, label in (("draw_sky", "Sky"), ("draw_shadows", "Shadows")):
            _, value = ui.checkbox(label, getattr(viewer.renderer, name))
            setattr(viewer.renderer, name, value)
        _, viewer.vsync = ui.checkbox("VSync", viewer.vsync)
        with workspace.widgets.group("Advanced", default_open=False) as opened:
            if opened:
                for callback in workspace.gui._ui_callbacks["rendering"]:
                    callback(ui)
                changed, paused = ui.checkbox("Freeze viewport image", viewer.is_rendering_paused())
                if changed:
                    viewer.set_rendering_paused(paused)
