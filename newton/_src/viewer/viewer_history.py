# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded display history recorded from the viewer's logging calls."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..sim.contacts import Contacts
from ..sim.control import Control
from ..sim.model import Model
from ..sim.state import State

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from newton.solvers import SolverBase

    from .viewer import ViewerBase


def _component_arrays(component: object | None, *, model: Model | None = None) -> dict[str, wp.array[Any]]:
    """Collect component arrays, optionally restricted to declared attributes."""
    if component is None:
        return {}
    arrays = {}
    for name, value in vars(component).items():
        if name.startswith("_"):
            continue
        if isinstance(value, wp.array):
            arrays[name] = value
        elif isinstance(value, Model.AttributeNamespace):
            for field_name, array in vars(value).items():
                if not field_name.startswith("_") and isinstance(array, wp.array):
                    arrays[f"{name}:{field_name}"] = array
    if model is not None:
        # Solvers can attach scratch arrays allocated inside CUDA graphs. Their
        # pointers need not be accessible outside graph execution. Only declared
        # State/Control attributes (including custom namespaces) are snapshots.
        arrays = {
            name: array
            for name, array in arrays.items()
            if model._attribute_spec(name) is not None or name in State.EXTENDED_ATTRIBUTES
        }
    return arrays


def _snapshot_array(array: wp.array[Any]) -> np.ndarray:
    """Retain independent host storage without recopying CUDA readbacks."""
    values = array.numpy()
    # CPU conversion aliases the live array; CUDA conversion already owns a host copy.
    return values.copy() if array.device.is_cpu else values


def _restore_component(component: object, template: object, fields: Mapping[str, np.ndarray]) -> None:
    """Upload recorded fields into display storage shaped like the logged component."""
    for qualified, values in fields.items():
        name = qualified
        target, source = component, template
        if ":" in name:
            namespace, name = name.split(":", 1)
            if not hasattr(target, namespace):
                setattr(target, namespace, Model.AttributeNamespace(namespace))
            target, source = getattr(target, namespace), getattr(source, namespace, None)
        original = getattr(source, name, None)
        if original is None:
            # The logged component no longer has this field; nothing can display it.
            continue
        # NumPy appends vector and matrix element dimensions to the storage shape.
        shape = values.shape[: original.ndim]
        array = getattr(target, name, None)
        if array is None or array.shape != shape or array.dtype != original.dtype:
            array = wp.empty(shape, dtype=original.dtype, device=original.device)
            setattr(target, name, array)
        array.assign(values)


@wp.kernel
def _count_self_contact_pairs(
    vertex_counts: wp.array[int],
    vertex_capacities: wp.array[int],
    edge_counts: wp.array[int],
    edge_capacities: wp.array[int],
    edge_offsets: wp.array[int],
    edge_pairs: wp.array[int],
    result: wp.array[int],
) -> None:
    i = wp.tid()
    if i < vertex_counts.shape[0]:
        wp.atomic_add(result, 0, wp.min(vertex_counts[i], vertex_capacities[i]))
    if i < edge_counts.shape[0]:
        count = int(0)
        for j in range(wp.min(edge_counts[i], edge_capacities[i])):
            offset = 2 * (edge_offsets[i] + j)
            # Edge pairs occur in both rows. Count each undirected pair once.
            if edge_pairs[offset] >= 0 and edge_pairs[offset] < edge_pairs[offset + 1]:
                count += 1
        wp.atomic_add(result, 1, count)


HistoryKey = tuple[str, str, "Hashable"]
"""``(call kind, viewer layer, call key)``; the key is a per-frame call ordinal or a logged object name."""


@dataclass
class RecordedArray:
    """Host copy of one logged Warp array, with what is needed to allocate its display storage."""

    values: np.ndarray
    dtype: Any
    device: wp.Device
    ndim: int

    @classmethod
    def snapshot(cls, array: wp.array[Any]) -> RecordedArray:
        """Copy a logged array to the host."""
        return cls(_snapshot_array(array), array.dtype, array.device, array.ndim)


@dataclass
class ContactRecord:
    """Active contact rows and solver observables of one :meth:`ViewerBase.log_contacts` call."""

    arrays: dict[str, np.ndarray]
    rigid_count: int
    soft_count: int
    observables: dict[str, np.ndarray] = field(default_factory=dict)
    state_key: HistoryKey | None = None
    """Key of a State logged in the same frame, when the contacts use that State."""
    state: dict[str, np.ndarray] = field(default_factory=dict)
    """Declared arrays of a separate contact State, such as the collision input poses."""
    observables_source: SolverBase.Observables | None = None
    """Logged observables container, whose solver allocates replay observables of the same kinds."""


@dataclass
class HistoryFrame:
    """Host-side display snapshot of one rendered frame, not a resumable solver checkpoint.

    Arrays are copied NumPy data keyed by attribute name (``namespace:name`` for
    custom attributes). Contact and observable arrays retain active rows only.
    """

    time: float
    """Simulation time passed to :meth:`ViewerBase.begin_frame` [s]."""
    states: dict[HistoryKey, dict[str, np.ndarray]] = field(default_factory=dict)
    """Declared arrays of each logged State."""
    contact_records: dict[HistoryKey, ContactRecord] = field(default_factory=dict)
    arrays: dict[HistoryKey, dict[str, Any]] = field(default_factory=dict)
    """Arguments of logged points and lines, keyed by their logged names.

    Warp arrays are :class:`RecordedArray` copies; ``None`` and uniform values
    are kept as passed, so replay reproduces cleared and uniform arguments.
    """
    control: dict[str, np.ndarray] = field(default_factory=dict)
    """Declared arrays of the inspected Control at the end of the frame."""
    size: int = 0
    """Bytes occupied by copied array data, excluding Python container overhead."""
    touching: int = 0
    body_speed: float = float("nan")
    particle_speed: float = float("nan")
    soft_shape_contacts: int = 0
    soft_self_contacts: int = 0

    @property
    def contacts(self) -> dict[str, np.ndarray]:
        """Return the recorded contact arrays of the frame's first log_contacts() call."""
        return next(iter(self.contact_records.values())).arrays if self.contact_records else {}

    @property
    def soft_touching(self) -> int:
        """Return the combined recorded soft-rigid and soft self-contact counts."""
        return self.soft_shape_contacts + self.soft_self_contacts


class FrameHistory:
    """Record bounded snapshots of what the application logs, and substitute them on replay.

    Between :meth:`begin_frame` and :meth:`end_frame`, the viewer passes the
    arguments of the application's own ``log_state()``, ``log_contacts()``,
    ``log_points()`` and ``log_lines()`` calls through :meth:`state`,
    :meth:`contacts` and :meth:`arrays`. While recording a new live frame, these
    copy the logged arrays to the host and return the live arguments. While
    replaying, they return display buffers holding the selected frame's data,
    matched to the same call by kind, layer, and per-frame ordinal or name.
    Calls without a recording pass through, so they show live data.

    The application's objects are never read outside those calls or modified.
    Replay does not rewind a solver: the live simulation stays at the newest
    frame and steps again only after playback reaches it. Memory and frame
    limits evict the oldest snapshots; ``index`` is relative to retained frames
    and ``first_frame`` counts evicted frames.
    """

    contact_fields = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1")
    soft_contact_fields = ("indices", "particle", "barycentric", "shape", "body_pos", "body_vel", "normal")

    def __init__(self, viewer: ViewerBase) -> None:
        """Initialize empty history for the viewer's current Model without recording a frame."""
        self.viewer = viewer
        self.model = viewer.model
        self.frames: list[HistoryFrame] = []
        self.index = 0
        self.first_frame = 0
        self.bytes_used = 0
        self.memory_limit = 128 * 2**20
        self.max_frames = 60000
        self.playback_speed = 1.0
        self._playback_fraction = 0.0
        self.recording = True
        self._recording_blocked = False
        self.message = ""
        self.dt = 1.0 / 60.0
        self.frame_open = False
        """True between begin_frame() and end_frame(), when application logging calls are recorded."""
        self.display_control: Control | None = None
        self._pending: HistoryFrame | None = None
        self._replay: HistoryFrame | None = None
        self._replace_next = False
        self._ordinals: dict[tuple[str, str], int] = {}
        self._logged_states: dict[int, HistoryKey] = {}
        self._display: dict[HistoryKey, Any] = {}
        self._loaded: dict[HistoryKey, int] = {}
        self._self_contact_counts = None
        self._plot_values = np.empty((self.max_frames + 64, 5), dtype=np.float64)
        self._plot_revision = 0
        self._plot_cache_key = None
        self._plot_cache = None

    @property
    def replaying(self) -> bool:
        """Report whether the selected retained frame precedes the newest live frame."""
        return bool(self.frames) and self.index < len(self.frames) - 1

    def select(self, index: int) -> None:
        """Clamp the requested retained-frame index; keep zero when history is empty."""
        self.index = max(0, min(int(index), len(self.frames) - 1))

    def advance(self, steps: float = 1) -> None:
        """Advance playback by speed-scaled steps, carrying fractional progress between calls."""
        self._playback_fraction += steps * self.playback_speed
        steps = int(self._playback_fraction)
        self._playback_fraction -= steps
        self.select(self.index + steps)

    def replace_next_frame(self) -> None:
        """Record the next frame even at an unchanged time, replacing the newest snapshot.

        Used after edits while paused, so the timeline shows the edited live state.
        """
        self._replace_next = True

    # -- frame boundaries ---------------------------------------------------------

    def begin_frame(self, time: float) -> None:
        """Start a replayed frame, a recorded live frame, or a pass-through frame at simulation time [s]."""
        self.frame_open = True
        self._ordinals.clear()
        self._logged_states.clear()
        self._pending = None
        self._replay = self.frames[self.index] if self.replaying else None
        if self._replay is not None or not self.recording or self._recording_blocked:
            return
        time = float(time)
        if not self.frames or time != self.frames[-1].time or self._replace_next:
            self._pending = HistoryFrame(time)

    def end_frame(self) -> None:
        """Commit a recorded frame, evicting old snapshots to stay within the memory and frame limits."""
        self.frame_open = False
        frame, self._pending, self._replay = self._pending, None, None
        if frame is None or not (frame.states or frame.contact_records or frame.arrays):
            return
        control = self.viewer._inspection_components.get("control")
        arrays = _component_arrays(control, model=self.model)
        if not self._reserve(frame, sum(a.capacity for a in arrays.values())):
            return
        frame.control = {name: _snapshot_array(array) for name, array in arrays.items()}
        frame.size += sum(a.nbytes for a in frame.control.values())
        replace_last = bool(self.frames) and frame.time == self.frames[-1].time
        self._replace_next = False
        if replace_last:
            self.bytes_used -= self.frames.pop().size
        self.frames.append(frame)
        self.record_plot_frame(frame)
        self.bytes_used += frame.size
        remove = 0
        while (self.bytes_used > self.memory_limit or len(self.frames) - remove > self.max_frames) and remove < len(
            self.frames
        ) - 1:
            self.bytes_used -= self.frames[remove].size
            remove += 1
        if remove:
            del self.frames[:remove]
            self.first_frame += remove
        if len(self.frames) > 1:
            self.dt = max(1e-9, self.frames[-1].time - self.frames[-2].time)
        self.index = len(self.frames) - 1
        self._loaded.clear()

    def _key(self, kind: str, name: Hashable | None = None) -> HistoryKey:
        """Identify a logging call by kind, active layer, and name or per-frame ordinal."""
        layer = self.viewer._active_layer_id
        if name is None:
            name = self._ordinals.get((kind, layer), 0)
            self._ordinals[kind, layer] = name + 1
        return kind, layer, name

    def _reserve(self, frame: HistoryFrame, size: int) -> bool:
        """Stop recording before a device-to-host transfer would exceed the per-frame snapshot limit."""
        if frame.size + size <= self.memory_limit // 2:
            return True
        self.message = "A frame exceeds the 64 MiB snapshot limit. Recording stopped."
        self._recording_blocked = True
        self._pending = None
        return False

    def _loaded_now(self, key: HistoryKey) -> bool:
        """Return whether a display buffer already holds the selected frame, marking it loaded."""
        if self._loaded.get(key) == self.index:
            return True
        self._loaded[key] = self.index
        return False

    # -- logging calls -------------------------------------------------------------

    def state(self, state: State) -> State:
        """Record a logged State, or return the display State of the replayed frame."""
        key = self._key("state")
        self._logged_states[id(state)] = key
        if self._pending is not None:
            self._record_state(key, state)
            return state
        if self._replay is None or key not in self._replay.states:
            return state
        return self._display_state(key, state, self._replay.states[key])

    def _record_state(self, key: HistoryKey, state: State) -> None:
        frame = self._pending
        arrays = _component_arrays(state, model=self.model)
        if not self._reserve(frame, sum(a.capacity for a in arrays.values())):
            return
        saved = {name: _snapshot_array(array) for name, array in arrays.items()}
        frame.states[key] = saved
        frame.size += sum(a.nbytes for a in saved.values())
        if len(frame.states) == 1:
            frame.body_speed = self.mean_speed(saved.get("body_qd"))
            frame.particle_speed = self.mean_speed(saved.get("particle_qd"))

    def _display_state(self, key: HistoryKey, template: State, fields: Mapping[str, np.ndarray]) -> State:
        display = self._display.get(key)
        if display is None:
            display = self._display[key] = State()
        if not self._loaded_now(key):
            _restore_component(display, template, fields)
        return display

    def contacts(
        self, contacts: Contacts, state: State, observables: SolverBase.Observables | None
    ) -> tuple[Contacts, State, SolverBase.Observables | None]:
        """Record logged contacts, or return display contacts, State and observables of the replayed frame."""
        key = self._key("contacts")
        if self._pending is not None:
            self._record_contacts(key, contacts, state, observables)
            return contacts, state, observables
        record = self._replay.contact_records.get(key) if self._replay is not None else None
        if record is None:
            return contacts, state, observables
        display = self._display.get(key)
        if (
            display is None
            or display.rigid_contact_max < contacts.rigid_contact_max
            or display.soft_contact_max < contacts.soft_contact_max
        ):
            display = self._display[key] = Contacts(
                contacts.rigid_contact_max, contacts.soft_contact_max, device=contacts.rigid_contact_count.device
            )
            self._loaded.pop(key, None)
            self._display.pop(("observables", *key[1:]), None)
        display_observables = self._display_observables(key, display, record) if record.observables else None
        if not self._loaded_now(key):
            display.rigid_contact_count.assign(np.array([record.rigid_count], dtype=np.int32))
            display.soft_contact_count.assign(np.array([record.soft_count], dtype=np.int32))
            for name, values in record.arrays.items():
                getattr(display, name)[: len(values)].assign(values)
            if display_observables is not None:
                for name, values in record.observables.items():
                    array = getattr(display_observables, name)
                    offset = 0
                    for start, n in self._observable_ranges(
                        display_observables, name, array, display, record.rigid_count, record.soft_count
                    ):
                        if n:
                            array[start : start + n].assign(values[offset : offset + n])
                        offset += n
        if record.state_key is not None and self._replay is not None and record.state_key in self._replay.states:
            state = self._display_state(record.state_key, state, self._replay.states[record.state_key])
        elif record.state:
            state = self._display_state(("contact_state", *key[1:]), state, record.state)
        return display, state, display_observables

    def _display_observables(
        self, key: HistoryKey, contacts: Contacts, record: ContactRecord
    ) -> SolverBase.Observables | None:
        """Allocate replay observables bound to the display contacts.

        The solver that produced the logged observables allocates them, so replay
        does not depend on the optional set_solver() binding.
        """
        observables_key = ("observables", *key[1:])
        display = self._display.get(observables_key)
        if display is not None:
            return display
        source = record.observables_source
        solver = source._solver
        if solver is None:
            return None
        display = solver.observables(kinds=source.kinds, contacts=contacts, requires_grad=False)
        self._display[observables_key] = display
        self._loaded.pop(key, None)
        return display

    def _record_contacts(
        self, key: HistoryKey, contacts: Contacts, state: State, observables: SolverBase.Observables | None
    ) -> None:
        frame = self._pending
        counters = contacts.contact_counters.numpy()
        count = min(int(counters[0]), contacts.rigid_contact_max)
        soft_count = min(int(counters[1]), contacts.soft_contact_max)
        arrays = {}
        for prefix, fields, n in (
            ("rigid_contact_", self.contact_fields, count),
            ("soft_contact_", self.soft_contact_fields, soft_count),
        ):
            for suffix in fields if n else ():
                array = getattr(contacts, prefix + suffix, None)
                if array is not None:
                    arrays[prefix + suffix] = (array, n)
        observable_arrays = {}
        if observables is not None and observables.model is self.model:
            for name, array in vars(observables).items():
                if not name.startswith("_") and isinstance(array, wp.array):
                    ranges = self._observable_ranges(observables, name, array, contacts, count, soft_count)
                    observable_arrays[name] = (array, ranges)
        state_key = self._logged_states.get(id(state))
        state_arrays = _component_arrays(state, model=self.model) if state_key is None else {}
        size = (
            sum(n * array.strides[0] for array, n in arrays.values())
            + sum(sum(n for _, n in ranges) * array.strides[0] for array, ranges in observable_arrays.values())
            + sum(a.capacity for a in state_arrays.values())
        )
        if not self._reserve(frame, size):
            return
        record = ContactRecord(
            {name: _snapshot_array(array[:n]) for name, (array, n) in arrays.items()},
            count,
            soft_count,
            {name: self._snapshot_observable(array, ranges) for name, (array, ranges) in observable_arrays.items()},
            state_key,
            {name: _snapshot_array(array) for name, array in state_arrays.items()},
            observables if observable_arrays else None,
        )
        frame.contact_records[key] = record
        frame.size += sum(
            a.nbytes for a in (*record.arrays.values(), *record.observables.values(), *record.state.values())
        )
        if len(frame.contact_records) == 1:
            frame.touching, frame.soft_shape_contacts = count, soft_count
            frame.soft_self_contacts = self._soft_self_contact_count(contacts) if self.model.particle_count else 0

    def arrays(self, kind: str, name: str, arrays: dict[str, Any]) -> dict[str, Any]:
        """Record the arguments of a named logging call, or substitute the replayed frame's arguments.

        Replay is independent of the live call's arguments: recorded arrays get
        their own display storage, and recorded ``None`` or uniform values replace
        the live ones.
        """
        key = self._key(kind, name)
        if self._pending is not None:
            live = [value for value in arrays.values() if isinstance(value, wp.array)]
            if self._reserve(self._pending, sum(a.capacity for a in live)):
                saved = {
                    arg: RecordedArray.snapshot(value) if isinstance(value, wp.array) else value
                    for arg, value in arrays.items()
                }
                self._pending.arrays[key] = saved
                self._pending.size += sum(v.values.nbytes for v in saved.values() if isinstance(v, RecordedArray))
            return arrays
        fields = self._replay.arrays.get(key) if self._replay is not None else None
        if fields is None:
            return arrays
        display = self._display.setdefault(key, {})
        loaded = self._loaded_now(key)
        result = dict(arrays)
        for arg, value in fields.items():
            if not isinstance(value, RecordedArray):
                result[arg] = value
                continue
            shape = value.values.shape[: value.ndim]
            array = display.get(arg)
            if array is None or array.shape != shape or array.dtype != value.dtype:
                array = display[arg] = wp.empty(shape, dtype=value.dtype, device=value.device)
                array.assign(value.values)
            elif not loaded:
                array.assign(value.values)
            result[arg] = array
        return result

    # -- recorded metrics ------------------------------------------------------------

    def _soft_self_contact_count(self, contacts: Contacts) -> int:
        """Count stored self-contact pairs without copying collision buffers to the CPU."""
        info = contacts.soft_self_contact_data
        if info is None:
            return 0
        if self._self_contact_counts is None:
            self._self_contact_counts = wp.zeros(2, dtype=int, device=self.model.device)
        else:
            self._self_contact_counts.zero_()
        wp.launch(
            _count_self_contact_pairs,
            dim=max(len(info.vertex_colliding_triangles_count), len(info.edge_colliding_edges_count)),
            inputs=[
                info.vertex_colliding_triangles_count,
                info.vertex_colliding_triangles_buffer_sizes,
                info.edge_colliding_edges_count,
                info.edge_colliding_edges_buffer_sizes,
                info.edge_colliding_edges_offsets,
                info.edge_colliding_edges,
            ],
            outputs=[self._self_contact_counts],
            device=self.model.device,
        )
        return int(self._self_contact_counts.numpy().sum())

    @staticmethod
    def _observable_ranges(
        observables: SolverBase.Observables,
        name: str,
        array: wp.array[Any],
        contacts: Contacts,
        count: int,
        soft_count: int,
    ) -> tuple[tuple[int, int], ...]:
        """Retain active contact rows while preserving each observable's row domain."""
        frequency = observables.get_attribute_frequency(name)
        if frequency == Model.AttributeFrequency.CONTACT:
            return ((0, count), (contacts.rigid_contact_max, soft_count))
        if frequency == Model.AttributeFrequency.CONTACT_RIGID:
            return ((0, count),)
        if frequency == Model.AttributeFrequency.CONTACT_SOFT:
            return ((0, soft_count),)
        return ((0, len(array)),)

    @staticmethod
    def _snapshot_observable(array: wp.array[Any], ranges: Sequence[tuple[int, int]]) -> np.ndarray:
        """Copy active (start, count) row ranges to one contiguous host snapshot.

        Preserve the trailing shape and dtype for empty selections. Reading
        each Warp slice synchronizes its device data; inactive capacity is not
        copied into history.
        """
        parts = [array[start : start + n].numpy() for start, n in ranges if n]
        # Warp rejects a zero-length slice starting at the array's end.
        # Preserve an empty array's dtype/shape without copying inactive rows.
        if not parts:
            return array[:0].numpy() if len(array) else array.numpy()
        return np.concatenate(parts)

    def restore_control(self) -> Control | None:
        """Return the selected frame's recorded Control, uploaded to display storage, or None."""
        frame = self.frames[self.index] if self.frames else None
        if frame is None or not frame.control:
            return None
        control = self.viewer._inspection_components.get("control")
        if control is None:
            return None
        if self.display_control is None:
            self.display_control = Control()
        if not self._loaded_now(("control", "", "")):
            _restore_component(self.display_control, control, frame.control)
        return self.display_control

    @staticmethod
    def mean_speed(velocity: ArrayLike | None) -> float:
        """Mean linear speed, using velocities already copied for history."""
        if velocity is None or len(velocity) == 0:
            return float("nan")
        linear = np.asarray(velocity)[:, :3]
        return float(np.sqrt(np.einsum("ij,ij->i", linear, linear)).mean(dtype=np.float64))

    def record_plot_frame(self, frame: HistoryFrame) -> None:
        """Store the latest frame contact counts and speeds in the fixed-size activity ring."""
        # A fixed ring keeps plot work independent of retained state size.
        index = (self.first_frame + len(self.frames) - 1) % len(self._plot_values)
        self._plot_values[index] = (
            frame.touching,
            frame.soft_shape_contacts,
            frame.soft_self_contacts,
            frame.body_speed,
            frame.particle_speed,
        )
        self._plot_revision += 1

    def plot_samples(self, low: int, high: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reduce an inclusive retained-frame range to approximately ``bins`` samples.

        Return bin centers, edges, three contact-count series, and two mean-speed
        series. Contacts use maxima to preserve brief impacts; motion uses means.
        All inputs come from the host-side plot ring, with no device readback.
        """
        key = (self.first_frame, low, high, bins, self._plot_revision)
        if self._plot_cache_key != key:
            indices = np.arange(low, high + 1)
            values = self._plot_values[(self.first_frame + indices) % len(self._plot_values)]
            # Anchor bins to absolute frame numbers, including through eviction.
            # Repartitioning the entire history on every append makes old peaks
            # flicker even though none of those recorded values have changed.
            stride = 1 if len(indices) <= bins else 1 << ((len(indices) - 1) // max(1, bins - 1)).bit_length()
            begin, end = self.first_frame + low, self.first_frame + high + 1
            boundaries = np.arange((begin // stride + 1) * stride, end, stride)
            edges = np.r_[begin, boundaries, end] - begin
            starts = edges[:-1]
            counts = np.diff(edges)
            contacts = np.maximum.reduceat(values[:, :3], starts, axis=0)
            speeds = np.add.reduceat(values[:, 3:], starts, axis=0) / counts[:, None]
            self._plot_cache = (low + (starts + edges[1:] - 1) * 0.5, low + edges, contacts, speeds)
            self._plot_cache_key = key
        return self._plot_cache

    def frame_at_time(self, time: float) -> int:
        """Return the first frame at or after time [s], clamped to the last; return -1 if empty."""
        return min(len(self.frames) - 1, bisect_left(self.frames, time, key=lambda frame: frame.time))

    def set_recording(self, enabled: bool) -> None:
        """Start a fresh bounded recording, or retain the current snapshots."""
        self.recording = enabled
        if enabled:
            self.message = ""
            self._recording_blocked = False
            self.frames.clear()
            self.index = self.bytes_used = self.first_frame = 0
            self._loaded.clear()
