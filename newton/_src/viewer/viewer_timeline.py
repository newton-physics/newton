# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local timeline presentation shared by the simulator workspace."""

from __future__ import annotations

from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from .viewer_widgets import PhysicsWidgets

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import ArrayLike

    from newton._src.viewer.viewer_history import HistoryFrame, WorkspaceHistory
    from newton.viewer import ViewerBase


class TimelineView:
    """Render contact and motion tracks with exact frame navigation."""

    def __init__(self, history: WorkspaceHistory) -> None:
        self.history = history

    @property
    def recording(self) -> bool:
        return self.history.recording

    @recording.setter
    def recording(self, value: bool) -> None:
        self.history.recording = value

    @property
    def playback_speed(self) -> float:
        return self.history.playback_speed

    @playback_speed.setter
    def playback_speed(self, value: float) -> None:
        self.history.playback_speed = value

    default_height = 58
    contact_log = True
    show_rigid_contacts = True
    show_soft_rigid_contacts = True
    show_soft_self_contacts = True
    show_body_speed = True
    show_particle_speed = True
    preference_names = (
        "recording",
        "playback_speed",
        "contact_log",
        "show_rigid_contacts",
        "show_soft_rigid_contacts",
        "show_soft_self_contacts",
        "show_body_speed",
        "show_particle_speed",
    )

    def contact_channels(self) -> list[tuple[str, str, str, int, tuple[float, float, float]]]:
        """Offer only contact families supported by the scene's entities."""
        channels = []
        if self.history.model.body_count:
            channels.append(("Rigid contacts", "touching", "show_rigid_contacts", 0, (0.40, 0.84, 0.78)))
        if self.history.model.particle_count:
            if self.history.model.shape_count:
                channels.append(
                    ("Soft-rigid contacts", "soft_shape_contacts", "show_soft_rigid_contacts", 1, (0.36, 0.65, 1.0))
                )
            channels.append(("Soft contacts", "soft_self_contacts", "show_soft_self_contacts", 2, (1.0, 0.50, 0.64)))
        return channels

    def _readout_frame(self, viewer: ViewerBase, *, now: float | None = None) -> tuple[int, HistoryFrame]:
        """Sample live numbers at 10 Hz, retaining an actual frame/time pair."""
        now = perf_counter() if now is None else now
        absolute = self.history.first_frame + self.history.index
        previous = getattr(self, "_readout", None)
        if (
            previous is None
            or viewer.is_paused()
            or self._readout_source is not self.history.frames
            or not self.history.first_frame <= previous[0] <= absolute
            or self.history.frames[self.history.index].time < previous[1].time
            or now - self._readout_time >= 0.1
        ):
            self._readout = (absolute, self.history.frames[self.history.index])
            self._readout_source = self.history.frames
            self._readout_time = now
        return self._readout

    def _plot_scales(
        self, contacts: float, speed: float, viewer: ViewerBase, *, now: float | None = None
    ) -> tuple[float, float]:
        """Damp live autoscaling without filtering recorded contact or motion data."""
        now = perf_counter() if now is None else now
        targets = np.array((np.log1p(contacts) if self.contact_log else contacts, speed))
        end = self.history.first_frame + len(self.history.frames)
        key = (id(self.history.frames), self.contact_log, *(getattr(self, name) for name in self.preference_names[3:]))
        if getattr(self, "_scale_key", None) != key or viewer.is_paused() or end < self._scale_end:
            self._scale_values = targets
        else:
            elapsed = max(0.0, now - self._scale_time)
            # Respond to new peaks quickly, then settle gently when peaks leave
            # the retained history. Time constants do not depend on render FPS.
            tau = np.where(targets > self._scale_values, 0.15, 0.8)
            self._scale_values += (targets - self._scale_values) * -np.expm1(-elapsed / tau)
        self._scale_key, self._scale_end, self._scale_time = key, end, now
        contact_scale, speed_scale = self._scale_values
        return (np.expm1(contact_scale) if self.contact_log else contact_scale), speed_scale

    def _draw_activity_track(self, ui: ModuleType, viewer: ViewerBase) -> None:
        scale = viewer.gui.ui.dpi_scale
        width = max(1, ui.get_content_region_avail().x)
        pos = ui.get_cursor_screen_pos()
        height = max(12 * scale, ui.get_content_region_avail().y)
        compact = height < 60 * scale
        draw = ui.get_window_draw_list()
        ui.invisible_button("##contact_track", ui.ImVec2(width, height))
        hovered, active = ui.is_item_hovered(), ui.is_item_active()
        if hovered or active:
            ui.set_mouse_cursor(ui.MouseCursor_.resize_ew)

        def color(rgba: ArrayLike) -> int:
            return ui.get_color_u32(ui.ImVec4(*rgba))

        corner = min(7 * scale, height / 2)
        end = ui.ImVec2(pos.x + width, pos.y + height)

        body_color = color((0.70, 0.61, 1.0, 1))
        particle_color = color((1.0, 0.72, 0.36, 1))
        muted = color((0.63, 0.63, 0.63, 1))
        grid = color((0.29, 0.29, 0.29, 1))
        draw.add_rect_filled(
            pos, end, color((0.24, 0.24, 0.24, 1) if hovered or active else (0.20, 0.20, 0.20, 1)), corner
        )
        x0, x1 = pos.x + 8 * scale, pos.x + width - 8 * scale
        y0 = pos.y + (2 if compact else 22) * scale
        y1 = pos.y + height - (2 if compact else 18) * scale
        plot_width, plot_height = max(1, x1 - x0), max(1, y1 - y0)
        count = len(self.history.frames)
        low, high = 0, count - 1
        if not ui.is_mouse_down(0):
            self._drag_span = None
        if getattr(self, "_drag_span", None) is not None:
            # Store absolute frames so a bounded history can evict old frames
            # while a drag is active without invalidating the visible range.
            low, high = (max(0, min(count - 1, i - self.history.first_frame)) for i in self._drag_span)
        extent = max(1, high - low)
        if ui.is_item_activated():
            self._drag_span = (self.history.first_frame + low, self.history.first_frame + high)
        ratio = np.clip((ui.get_io().mouse_pos.x - x0) / plot_width, 0, 1)
        # A one-frame history has no index interval, even though its drawing
        # extent is nonzero. Both hover and scrub must select frame zero.
        hover_index = low + round(float(ratio) * (high - low))
        if active and ui.is_mouse_down(0):
            viewer._paused = True
            viewer._step_requested = False
            self.history.select(hover_index)
        bins = min(512, max(2, int(plot_width / (2 * scale))))
        samples, edges, contacts, speeds = self.history.plot_samples(low, high, bins)
        channels = [channel for channel in self.contact_channels() if getattr(self, channel[2])]
        max_contacts = max((float(contacts[:, channel[3]].max()) for channel in channels), default=0.0)
        peak_contacts = max(1.0, max_contacts)
        visible_speeds = speeds.copy()
        if not self.show_body_speed:
            visible_speeds[:, 0] = np.nan
        if not self.show_particle_speed:
            visible_speeds[:, 1] = np.nan
        finite = visible_speeds[np.isfinite(visible_speeds)]
        peak_speed = float(finite.max()) if finite.size else 0.0
        speed_scale = peak_speed if peak_speed > 1e-9 else 1.0
        peak_contacts, speed_scale = self._plot_scales(peak_contacts, speed_scale, viewer)
        _, frame = self._readout_frame(viewer)
        legends = []
        for label, attr, _, _, rgb in channels:
            legends.append((f"{label}  {getattr(frame, attr, 0):,}", color((*rgb, 0.90))))
        for label, attr, enabled, tint in (
            ("Body", "body_speed", self.show_body_speed, body_color),
            ("Particles", "particle_speed", self.show_particle_speed, particle_color),
        ):
            value = getattr(frame, attr, float("nan"))
            if enabled and np.isfinite(value):
                legends.append((f"{label}  {value:.3g} m/s", tint))
        legend_x = pos.x + 8 * scale
        for text, tint in () if compact else legends:
            if legend_x + ui.calc_text_size(text).x < pos.x + width - 8 * scale:
                draw.add_text(ui.ImVec2(legend_x, pos.y + 2 * scale), tint, text)
                legend_x += ui.calc_text_size(text).x + 22 * scale
        for ratio in () if compact else (0.0, 0.5, 1.0):
            y = y1 - ratio * plot_height
            draw.add_line(ui.ImVec2(x0, y), ui.ImVec2(x1, y), grid)
        ticks = max(1, min(10, int(plot_width / (90 * scale)), extent))
        for tick in () if compact else range(ticks + 1):
            x = x0 + plot_width * tick / ticks
            draw.add_line(ui.ImVec2(x, y0), ui.ImVec2(x, y1), grid)
            t = self.history.frames[min(high, low + round(extent * tick / ticks))].time
            label = f"{t:.3f}"
            draw.add_text(ui.ImVec2(min(x, x1 - ui.calc_text_size(label).x), y1 + 3 * scale), muted, label)
        draw.push_clip_rect(ui.ImVec2(x0, y0 - 2 * scale), ui.ImVec2(x1 + 1, y1 + 1), True)
        for _, _, _, column, rgb in channels:
            counts = contacts[:, column]
            heights = np.log1p(counts) / np.log1p(peak_contacts) if self.contact_log else counts / peak_contacts
            heights = np.clip(heights, 0, 1)
            xs = x0 + plot_width * np.minimum(edges - low, extent) / extent
            # Share fill edges exactly, with no antialiased internal borders.
            # Merge equal neighbors and stroke only the continuous step outline.
            runs = np.r_[0, np.flatnonzero(np.diff(counts)) + 1, len(counts)]
            outline = []
            for start, stop in pairwise(runs):
                left, right = float(xs[start]), float(xs[stop])
                y = y1 - float(heights[start]) * plot_height
                if heights[start] > 0 and right > left:
                    draw.add_rect_filled(ui.ImVec2(left, y), ui.ImVec2(right, y1), color((*rgb, 0.24)))
                outline.append(ui.ImVec2(left, y))
                if right > left:
                    outline.append(ui.ImVec2(right, y))
            draw.add_polyline(outline, color((*rgb, 0.90)), 1.25 * scale, 0)
        for column, tint in enumerate((body_color, particle_color)):
            previous = None
            for j, speed in zip(samples, visible_speeds[:, column], strict=True):
                if not np.isfinite(speed):
                    previous = None
                    continue
                point = ui.ImVec2(x0 + plot_width * (j - low) / extent, y1 - min(1, speed / speed_scale) * plot_height)
                if previous is not None:
                    draw.add_line(previous, point, tint, 1.6 * scale)
                previous = point
        draw.pop_clip_rect()
        draw.add_rect(
            pos, end, color((0.48, 0.48, 0.48, 1) if hovered or active else (0.30, 0.30, 0.30, 1)), corner, scale
        )
        x = x0 + plot_width * (self.history.index - low) / extent
        accent = (1.0, 0.73, 0.30)
        top, bottom = pos.y + scale, end.y - scale
        for thickness, alpha in ((8, 0.10), (4, 0.24), (1.75, 1.0)):
            draw.add_line(ui.ImVec2(x, top), ui.ImVec2(x, bottom), color((*accent, alpha)), thickness * scale)
        tip = min(5 * scale, height * 0.28)
        for base, direction in ((top, 1), (bottom, -1)):
            draw.add_triangle_filled(
                ui.ImVec2(x - 4 * scale, base),
                ui.ImVec2(x + 4 * scale, base),
                ui.ImVec2(x, base + direction * tip),
                color((*accent, 1)),
            )
        if hovered:
            hover = self.history.frames[hover_index]
            x = x0 + plot_width * (hover_index - low) / extent
            if hover_index != self.history.index:
                draw.add_line(ui.ImVec2(x, y0), ui.ImVec2(x, y1), muted)
            details = [f"Frame {self.history.first_frame + hover_index}  |  {hover.time:.6f} s"]
            for label, attr, _, _, _ in self.contact_channels():
                details.append(f"{label}: {getattr(hover, attr, 0):,}")
            for label, attr in (("Mean body speed", "body_speed"), ("Mean particle speed", "particle_speed")):
                value = getattr(hover, attr, float("nan"))
                if np.isfinite(value):
                    details.append(f"{label}: {value:.6g} m/s")
            details.append("Drag to scrub")
            ui.begin_tooltip()
            ui.push_text_wrap_pos(360 * scale)
            ui.text_unformatted("\n".join(details))
            ui.pop_text_wrap_pos()
            ui.end_tooltip()

    def draw_gui(self, ui: ModuleType, viewer: ViewerBase, *, embedded: bool = False) -> None:
        """Draw a scrub-enabled track beside the editable frame and time."""
        scale = viewer.gui.ui.dpi_scale
        if not embedded:
            screen = ui.get_io().display_size
            ui.set_next_window_pos(ui.ImVec2(0, screen.y - self.default_height * scale))
            ui.set_next_window_size(ui.ImVec2(screen.x, self.default_height * scale))
            ui.begin("Trajectory timeline", flags=ui.WindowFlags_.no_title_bar)
        widgets = PhysicsWidgets(viewer.gui)
        if ui.begin_table("##timeline_layout", 2, ui.TableFlags_.sizing_stretch_prop):
            ui.table_setup_column("Track", ui.TableColumnFlags_.width_stretch)
            ui.table_setup_column(
                "Readouts",
                ui.TableColumnFlags_.width_fixed,
                min(210 * scale, ui.get_content_region_avail().x * 0.28),
            )
            ui.table_next_column()
            self._draw_transport(ui, viewer, widgets)
            if self.history.frames:
                self._draw_activity_track(ui, viewer)
            ui.table_next_column()
            ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(8 * scale, 2 * scale))
            ui.push_style_var(ui.StyleVar_.frame_padding, ui.ImVec2(5 * scale, scale))
            ui.push_style_var(ui.StyleVar_.item_spacing, ui.ImVec2(6 * scale, 3 * scale))
            if ui.begin_child(
                "##timeline_readouts",
                ui.ImVec2(0, 0),
                ui.ChildFlags_.always_use_window_padding,
            ):
                pos = ui.get_window_pos()
                ui.get_window_draw_list().add_line(
                    pos,
                    ui.ImVec2(pos.x, pos.y + ui.get_window_size().y),
                    ui.get_color_u32(ui.ImVec4(0.34, 0.34, 0.34, 1)),
                )
                self._draw_readouts(ui, viewer, widgets)
            ui.end_child()
            ui.pop_style_var(3)
            ui.end_table()
        if not embedded:
            ui.end()

    def _draw_transport(self, ui: ModuleType, viewer: ViewerBase, widgets: PhysicsWidgets) -> None:
        scale = viewer.gui.ui.dpi_scale
        row_y = ui.get_cursor_pos_y()
        ui.push_style_var(ui.StyleVar_.frame_padding, ui.ImVec2(6 * scale, (22 * scale - ui.get_font_size()) / 2))
        if widgets.icon_button(
            "##timeline_play",
            "play" if viewer.is_paused() else "pause",
            "Play from selected frame" if viewer.is_paused() else "Pause",
            size=22,
            prominent=True,
        ):
            viewer._paused = not viewer.is_paused()
            viewer._step_requested = False
        ui.same_line(spacing=4 * scale)
        ui.set_cursor_pos_y(row_y)
        if widgets.icon_button(
            "##timeline_start",
            "first",
            "First recorded frame",
            size=22,
            enabled=bool(self.history.frames) and self.history.index > 0,
            show_lock=False,
        ):
            viewer._paused, viewer._step_requested = True, False
            self.history.select(0)
        for label, icon, delta in (
            ("##previous_timeline_frame", "previous", -1),
            ("##next_timeline_frame", "play", 1),
        ):
            ui.same_line(spacing=4 * scale)
            ui.set_cursor_pos_y(row_y)
            enabled = 0 <= self.history.index + delta < len(self.history.frames) if self.history.frames else delta > 0
            if widgets.icon_button(
                label,
                icon,
                "Previous physics frame" if delta < 0 else "Next physics frame",
                size=22,
                enabled=enabled,
                show_lock=False,
            ):
                viewer._paused = True
                viewer._step_requested = not self.history.frames
                if self.history.frames:
                    self.history.select(self.history.index + delta)
        ui.same_line(spacing=4 * scale)
        ui.set_cursor_pos_y(row_y)
        if widgets.icon_button(
            "##timeline_end",
            "last",
            "Latest recorded frame · resume live simulation here",
            size=22,
            enabled=bool(self.history.frames) and self.history.index < len(self.history.frames) - 1,
            show_lock=False,
        ):
            viewer._paused, viewer._step_requested = True, False
            self.history.select(len(self.history.frames) - 1)
        ui.same_line(spacing=6 * scale)
        ui.set_cursor_pos_y(row_y)
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
        ui.set_next_item_width(75 * scale)
        changed, speed = ui.combo("##playback_speed", speeds.index(self.playback_speed), [f"{x:g}x" for x in speeds])
        if changed:
            self.playback_speed = speeds[speed]
        ui.same_line(spacing=10 * scale)
        ui.set_cursor_pos_y(row_y)
        self.draw_recording_control(ui)
        ui.pop_style_var()
        if self.history.message:
            ui.text_wrapped(self.history.message)

    def _draw_readouts(self, ui: ModuleType, viewer: ViewerBase, widgets: PhysicsWidgets) -> None:
        if not self.history.frames:
            ui.text_disabled("Frame  —")
            ui.text_disabled("Time   —")
            return
        count = len(self.history.frames)
        absolute, frame = self._readout_frame(viewer)
        edit = widgets.number(
            "Frame",
            absolute,
            editable=True,
            integer=True,
            speed=1,
            bounds=(self.history.first_frame, self.history.first_frame + count - 1),
            label_width=60 * widgets.scale,
        )
        if edit.activated or edit.changed:
            viewer._paused = True
            viewer._step_requested = False
            self.history.select(edit.value - self.history.first_frame)
            absolute, frame = self._readout_frame(viewer)
        time = widgets.number(
            "Time",
            frame.time,
            editable=True,
            speed=self.history.dt,
            bounds=(self.history.frames[0].time, self.history.frames[-1].time),
            unit="s",
            label_width=60 * widgets.scale,
            format_spec=".6f" if viewer.is_paused() else ".3f",
        )
        if time.activated or time.changed:
            viewer._paused = True
            viewer._step_requested = False
            self.history.select(
                self.history.frame_at_time(time.value) if time.changed else absolute - self.history.first_frame
            )

    def status_items(self) -> list[tuple[str, str, tuple[float, float, float]]]:
        """Read-only metrics for the selected frame, without device transfers."""
        neutral = (0.82, 0.82, 0.82)

        def count(value: int) -> str:
            if value < 10_000:
                return f"{value:,}"
            for power, suffix in ((12, "T"), (9, "B"), (6, "M"), (3, "k")):
                if value >= 10**power:
                    return f"{value / 10**power:.1f}{suffix}"

        items = [("Step", f"{self.history.dt * 1000:.2f} ms", neutral)]
        if self.history.frames:
            frame = self.history.frames[self.history.index]
            for label, attr, _, _, rgb in self.contact_channels():
                items.append((label, count(getattr(frame, attr, 0)), rgb))
            if not hasattr(self.history, "recording"):
                items.append(("Depth", f"{frame.depth_mm:.2f} mm", neutral))
        memory = (
            f"{self.history.bytes_used / 2**20:.1f} MiB"
            if self.history.bytes_used < 2**30
            else f"{self.history.bytes_used / 2**30:.1f} GiB"
        )
        items.extend((("States", count(len(self.history.frames)), neutral), ("History", memory, neutral)))
        if self.history.frames:
            for label, attr, entities, rgb in (
                ("Body speed", "body_speed", self.history.model.body_count, (0.70, 0.61, 1.0)),
                ("Particle speed", "particle_speed", self.history.model.particle_count, (1.0, 0.72, 0.36)),
            ):
                if not entities:
                    continue
                value = getattr(frame, attr, float("nan"))
                if np.isfinite(value):
                    speed = f"{value:.2f}" if abs(value) < 10_000 else f"{value:.2e}"
                    items.append((label, f"{speed} m/s", rgb))
                else:
                    items.append((label, "—", rgb))
        return items

    def draw_recording_control(self, ui: ModuleType) -> None:
        changed, enabled = ui.checkbox("Record history", self.recording)
        if changed:
            self.history.set_recording(enabled)
        if ui.is_item_hovered():
            ui.set_tooltip(
                "Record state and contacts in a bounded 128 MiB history. Playback continues simulation at the live frame."
            )
