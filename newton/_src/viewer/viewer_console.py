# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded logging capture and a virtualized console for the local viewer."""

import logging
import re
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import ClassVar

from ..exceptions import NewtonDeprecationWarning, NewtonWarning

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_LEVELS = ((10, "Debug"), (20, "Info"), (30, "Warning"), (40, "Error"), (50, "Critical"))


@dataclass(frozen=True, slots=True)
class ConsoleEntry:
    sequence: int
    timestamp: str
    level: int
    severity: str
    source: str
    location: str
    message: str
    lines: tuple[str, ...]
    search: str

    def text(self):
        return f"{self.timestamp} {self.severity:<8} {self.source}: {self.message}"


class ConsoleBuffer(logging.Handler):
    """Capture text on producer threads without touching ImGui or retaining tracebacks."""

    def __init__(self, *, capacity=4000, max_characters=4 * 1024 * 1024, max_lines=20000):
        super().__init__(logging.DEBUG)
        self.entries = deque()
        self.capacity = capacity
        self.max_characters = max_characters
        self.max_lines = max_lines
        self.characters = self.line_count = self.discarded = self.sequence = self.revision = 0
        self.warning_sequence = 0
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            # Format exceptions now; retaining LogRecords would retain entire simulation frames.
            message = _ANSI.sub("", self.format(record)).replace("\r\n", "\n").replace("\r", "\n")
            message = message.replace("\x00", "").expandtabs(4)
            if len(message) > 16384:
                message = message[:16384] + "\n… message truncated"
            lines = message.splitlines() or [""]
            if len(lines) > 256:
                lines = [*lines[:255], "… message truncated"]
            message = "\n".join(lines)
            source = record.name
            stamp = time.strftime("%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}"
            self.sequence += 1
            entry = ConsoleEntry(
                self.sequence,
                stamp,
                record.levelno,
                record.levelname,
                source,
                f"{record.pathname}:{record.lineno}",
                message,
                tuple(lines),
                f"{source} {record.levelname} {message}".casefold(),
            )
            self.entries.append(entry)
            if record.levelno >= logging.WARNING:
                self.warning_sequence = self.sequence
            self.characters += len(message)
            self.line_count += len(lines)
            while self.entries and (
                len(self.entries) > self.capacity
                or self.characters > self.max_characters
                or self.line_count > self.max_lines
            ):
                old = self.entries.popleft()
                self.characters -= len(old.message)
                self.line_count -= len(old.lines)
                self.discarded += 1
            self.revision += 1
        except Exception:
            self.handleError(record)

    def snapshot(self):
        self.acquire()
        try:
            return self.revision, tuple(self.entries), self.discarded
        finally:
            self.release()

    def clear(self):
        self.acquire()
        try:
            self.entries.clear()
            self.characters = self.line_count = self.discarded = 0
            self.warning_sequence = 0
            self.revision += 1
        finally:
            self.release()


class ConsoleCapture:
    """Own the GUI sink and scoped Newton warning routing across viewer lifetimes."""

    _active: ClassVar[set] = set()
    _previous_warning = None
    _warning_hook = None
    _previous_level = None
    _owned_level = None

    def __init__(self):
        self.buffer = ConsoleBuffer()
        self.logger = logging.getLogger("newton")
        self.closed = False
        cls = type(self)
        if not cls._active:
            cls._previous_level = self.logger.level
            cls._owned_level = None
            cls._previous_warning = warnings.showwarning
            cls._warning_hook = cls._make_warning_hook(warnings.showwarning)
            warnings.showwarning = cls._warning_hook
        cls._active.add(self)
        self.logger.addHandler(self.buffer)

    @classmethod
    def _make_warning_hook(cls, previous):
        def showwarning(message, category, filename, lineno, file=None, line=None):
            # Python has already applied its warning filters before reaching this hook.
            if (
                cls._active
                and cls._warning_hook is showwarning
                and file is None
                and issubclass(category, (NewtonWarning, NewtonDeprecationWarning))
            ):
                logger = logging.getLogger("newton.warnings")
                if logger.isEnabledFor(logging.WARNING):
                    record = logger.makeRecord(
                        logger.name, logging.WARNING, filename, lineno, "%s: %s", (category.__name__, message), None
                    )
                    logger.handle(record)
            else:
                previous(message, category, filename, lineno, file=file, line=line)

        return showwarning

    def request_level(self, level):
        # Display filtering never discards already captured records. Enable extra
        # verbosity only when the user explicitly asks to see it.
        if self.logger.getEffectiveLevel() > level:
            self.logger.setLevel(level)
            type(self)._owned_level = level

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.logger.removeHandler(self.buffer)
        self.buffer.close()
        cls = type(self)
        cls._active.discard(self)
        if not cls._active:
            if warnings.showwarning is cls._warning_hook:
                warnings.showwarning = cls._previous_warning
            cls._warning_hook = None
            if cls._owned_level is not None and self.logger.level == cls._owned_level:
                self.logger.setLevel(cls._previous_level)
            cls._owned_level = None


class ViewerConsole:
    """Present a log drawer without scanning or laying out hidden rows every frame."""

    def __init__(self, capture):
        self.capture = capture
        self.open = False
        self.level = logging.INFO
        self.search = ""
        self.follow = True
        self.selected = None
        self._context_entry = None
        self.entries = ()
        self.rows = []
        self._longest_line = 0
        self.discarded = 0
        self._key = None
        self._refresh_time = 0.0
        self._scroll_to_end = True
        self._scroll_y = 0.0
        self._last_seen = 0
        self._last_warning = 0
        self._rects = {}

    def poll_warnings(self):
        """Open once per new warning, on the UI thread and without scanning history."""
        sequence = self.capture.buffer.warning_sequence
        if sequence <= self._last_warning:
            return False
        self._last_warning = sequence
        if not self.open:
            # Do not reveal an empty drawer because a previous filter hid the warning.
            self.level = min(self.level, logging.WARNING)
            self.search = ""
            self._scroll_to_end = True
            self.refresh(force=True)
        self.open = True
        return True

    def refresh(self, *, force=False):
        now = time.perf_counter()
        buffer = self.capture.buffer
        key = (buffer.revision, self.level, self.search)
        if key == self._key or (
            not force and key[1:] == (self._key or (None, None, None))[1:] and now - self._refresh_time < 0.1
        ):
            return
        revision, entries, self.discarded = buffer.snapshot()
        query = self.search.casefold().strip()
        self.entries = tuple(e for e in entries if e.level >= self.level and (not query or query in e.search))
        self.rows = [(entry, line) for entry in self.entries for line in range(len(entry.lines))]
        self._longest_line = max((len(entry.lines[line]) for entry, line in self.rows), default=0)
        self._scroll_to_end |= self.follow
        self._key = revision, self.level, self.search
        self._refresh_time = now

    @staticmethod
    def color(level):
        if level >= logging.ERROR:
            return (1.0, 0.43, 0.43, 1.0)
        if level >= logging.WARNING:
            return (0.96, 0.74, 0.35, 1.0)
        if level >= logging.INFO:
            return (0.85, 0.85, 0.85, 1.0)
        return (0.58, 0.58, 0.58, 1.0)

    def toggle(self):
        self.open = not self.open
        if self.open:
            self._scroll_to_end = self.follow

    def status_button(self, workspace):
        ui, widgets = workspace.gui.ui.imgui, workspace.widgets
        s = widgets.scale
        # Fit the status row itself, independently of the window padding or
        # the previous item's baseline. Reserve a stable unread-count slot.
        pos = ui.get_cursor_screen_pos()
        top, height = ui.get_window_pos().y, ui.get_window_size().y
        width = (24 + 55) * s + ui.calc_text_size("Console").x
        ui.set_cursor_screen_pos(ui.ImVec2(pos.x, top + 2 * s))
        if ui.invisible_button("##console_toggle", ui.ImVec2(width, height - 4 * s)):
            self.toggle()
        self._rects["toggle"] = (tuple(ui.get_item_rect_min()), tuple(ui.get_item_rect_max()))
        draw = ui.get_window_draw_list()
        if ui.is_item_hovered() or self.open:
            draw.add_rect_filled(
                ui.get_item_rect_min(), ui.get_item_rect_max(), widgets.color((0.22, 0.22, 0.22, 1)), 3 * s
            )
        widgets.icon("console", (pos.x, top + (height - 16 * s) / 2), 16 * s, active=self.open)
        text_y = widgets.text_top(top, height)
        draw.add_text(
            ui.ImVec2(pos.x + 22 * s, text_y), widgets.color(widgets.accent if self.open else widgets.muted), "Console"
        )
        unread = max(0, self.capture.buffer.sequence - self._last_seen) if not self.open else 0
        if unread:
            label = f"{min(unread, 999)}{'+' if unread > 999 else ''} new"
            draw.add_text(
                ui.ImVec2(pos.x + 28 * s + ui.calc_text_size("Console").x, text_y), widgets.color(widgets.accent), label
            )
        if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            ui.set_tooltip(f"Show or hide console (Ctrl+`)\n{unread:,} unread log entries.")
        ui.same_line(pos.x - ui.get_window_pos().x + width + 18 * s)

    def draw(self, workspace):
        ui, s = workspace.gui.ui.imgui, workspace.widgets.scale
        self.refresh()
        self._last_seen = self.capture.buffer.sequence
        # One compact toolbar with a stable baseline at every DPI.
        row_y = ui.get_cursor_pos_y()

        def next_control():
            ui.same_line(spacing=6 * s)
            ui.set_cursor_pos_y(row_y)

        ui.push_style_var(ui.StyleVar_.frame_padding, ui.ImVec2(6 * s, max(1, (24 * s - ui.get_font_size()) / 2)))
        workspace.widgets.label("Console", height=24 * s)
        next_control()
        ui.set_next_item_width(110 * s)
        current = next(name for level, name in _LEVELS if level == self.level)
        expanded = ui.begin_combo("##console_level", current)
        self._rects["level"] = (tuple(ui.get_item_rect_min()), tuple(ui.get_item_rect_max()))
        if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            ui.set_tooltip("Minimum severity to display")
        if expanded:
            for level, name in _LEVELS:
                ui.push_style_color(ui.Col_.text, ui.ImVec4(*self.color(level)))
                clicked, _ = ui.selectable(name, level == self.level)
                ui.pop_style_color()
                self._rects[f"level_{level}"] = (tuple(ui.get_item_rect_min()), tuple(ui.get_item_rect_max()))
                if clicked:
                    self.level = level
                    self.capture.request_level(level)
                    self.refresh(force=True)
                if level == self.level:
                    ui.set_item_default_focus()
            ui.end_combo()
        next_control()
        ui.set_next_item_width(max(80 * s, min(300 * s, ui.get_content_region_avail().x - 205 * s)))
        changed, self.search = ui.input_text_with_hint("##console_search", "Filter messages or source", self.search)
        self._rects["search"] = (tuple(ui.get_item_rect_min()), tuple(ui.get_item_rect_max()))
        if changed:
            self.refresh(force=True)
        next_control()
        if workspace.widgets.icon_button(
            "##console_follow", "last", "Follow newest messages", active=self.follow, size=24
        ):
            self.follow = not self.follow
            self._scroll_to_end = self.follow
        next_control()
        if workspace.widgets.icon_button("##console_copy", "copy", "Copy filtered log", size=24):
            ui.set_clipboard_text("\n".join(entry.text() for entry in self.entries))
        next_control()
        if workspace.widgets.icon_button("##console_clear", "clear", "Clear console history", size=24):
            self.capture.buffer.clear()
            self.selected = None
            self.refresh(force=True)
        next_control()
        workspace.widgets.label(f"{len(self.entries):,}", height=24 * s, width=50 * s, muted=True)
        if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            ui.set_tooltip(f"Matching records; {self.discarded:,} older records discarded by the history limit")
        next_control()
        ui.set_cursor_pos_x(max(ui.get_cursor_pos_x(), ui.get_window_width() - 34 * s))
        if workspace.widgets.icon_button("##console_close", "close", "Hide console", size=24):
            self.open = False
        self._rects["close"] = (tuple(ui.get_item_rect_min()), tuple(ui.get_item_rect_max()))
        ui.pop_style_var()

        font = workspace.gui.ui.number_font
        if font is not None:
            ui.push_font(font, 13.0)
        ui.push_style_var(ui.StyleVar_.cell_padding, ui.ImVec2(7 * s, 2 * s))
        flags = (
            ui.TableFlags_.scroll_y
            | ui.TableFlags_.scroll_x
            | ui.TableFlags_.row_bg
            | ui.TableFlags_.resizable
            | ui.TableFlags_.sizing_fixed_fit
            | ui.TableFlags_.no_saved_settings
        )
        height = max(1, ui.get_content_region_avail().y)
        message_width = max(
            ui.get_content_region_avail().x - 434 * s,
            self._longest_line * ui.calc_text_size("M").x + 20 * s,
        )
        if ui.begin_table("##console_records", 4, flags, ui.ImVec2(0, height)):
            ui.table_setup_column("Time", ui.TableColumnFlags_.width_fixed, 106 * s)
            ui.table_setup_column("Level", ui.TableColumnFlags_.width_fixed, 74 * s)
            ui.table_setup_column(
                "Source", ui.TableColumnFlags_.width_fixed, min(220 * s, ui.get_window_width() * 0.22)
            )
            ui.table_setup_column(
                "Message", ui.TableColumnFlags_.width_fixed | ui.TableColumnFlags_.no_resize, message_width
            )
            ui.table_setup_scroll_freeze(3, 1)
            ui.table_headers_row()
            # A scroll gesture takes precedence over auto-follow, even during a burst.
            hovered = ui.is_window_hovered(ui.HoveredFlags_.child_windows)
            scroll = ui.get_scroll_y()
            if hovered and (ui.get_io().mouse_wheel > 0 or (ui.is_mouse_down(0) and scroll < self._scroll_y - 1)):
                self.follow = self._scroll_to_end = False
            row_height = ui.get_text_line_height() + 4 * s
            clipper = ui.ListClipper()
            clipper.begin(len(self.rows), row_height)
            context_requested = False
            while clipper.step():
                for row in range(clipper.display_start, clipper.display_end):
                    entry, line = self.rows[row]
                    ui.table_next_row(0, row_height)
                    ui.table_set_column_index(0)
                    ui.push_id(f"{entry.sequence}:{line}")
                    ui.push_style_color(ui.Col_.text, ui.ImVec4(0.58, 0.58, 0.58, 1))
                    clicked, _ = ui.selectable(
                        (entry.timestamp if line == 0 else "") + "##record",
                        self.selected == entry.sequence,
                        ui.SelectableFlags_.span_all_columns,
                    )
                    ui.pop_style_color()
                    if clicked:
                        self.selected = entry.sequence
                    if ui.is_item_hovered() and ui.is_mouse_released(ui.MouseButton_.right):
                        self.selected = entry.sequence
                        self._context_entry = entry
                        context_requested = True
                    if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
                        ui.begin_tooltip()
                        ui.push_text_wrap_pos(520 * s)
                        ui.text_disabled(entry.location)
                        preview = "\n".join(entry.lines[:12])[:1600]
                        ui.text_unformatted(preview + ("\n…" if preview != entry.message else ""))
                        ui.pop_text_wrap_pos()
                        ui.end_tooltip()
                    ui.table_set_column_index(1)
                    ui.text_colored(ui.ImVec4(*self.color(entry.level)), entry.severity if line == 0 else "")
                    ui.table_set_column_index(2)
                    ui.text_disabled(entry.source.removeprefix("newton._src.") if line == 0 else "")
                    ui.table_set_column_index(3)
                    ui.text_colored(ui.ImVec4(*self.color(entry.level)), entry.lines[line])
                    ui.pop_id()
            clipper.end()
            if self._scroll_to_end or self.follow:
                ui.set_scroll_y(len(self.rows) * row_height)
                self._scroll_to_end = False
            self._scroll_y = ui.get_scroll_y()
            self._scroll_max = ui.get_scroll_max_y()
            self._rects["records"] = (tuple(ui.get_window_pos()), tuple(ui.get_window_size()))
            ui.end_table()
            if context_requested:
                ui.open_popup("##record_actions")
        ui.pop_style_var()
        if font is not None:
            ui.pop_font()
        # Keep the popup outside the clipped rows. Its immutable record remains
        # valid when live logging scrolls the row away or evicts it from history.
        if ui.begin_popup("##record_actions"):
            try:
                entry = self._context_entry
                if entry is not None:
                    if ui.menu_item_simple("Copy message"):
                        ui.set_clipboard_text(entry.message)
                    if ui.menu_item_simple("Copy record", "Ctrl+C"):
                        ui.set_clipboard_text(entry.text())
            finally:
                ui.end_popup()
        else:
            self._context_entry = None
        if (
            ui.is_window_focused(ui.FocusedFlags_.root_and_child_windows)
            and not ui.get_io().want_text_input
            and ui.get_io().key_ctrl
            and ui.is_key_pressed(ui.Key.c, False)
        ):
            selected = self._context_entry or next(
                (entry for entry in self.entries if entry.sequence == self.selected), None
            )
            if selected is not None:
                ui.set_clipboard_text(selected.text())
