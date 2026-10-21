# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded render-frame history for conventional Newton examples."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..sim.contacts import Contacts
from ..sim.model import Model
from ..sim.state import State

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from newton._src.viewer.viewer_types import ViewerExample
    from newton.solvers import SolverBase


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
            for field, array in vars(value).items():
                if not field.startswith("_") and isinstance(array, wp.array):
                    arrays[f"{name}:{field}"] = array
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


def _restore_component(component: object, live: object, fields: dict[str, np.ndarray]) -> None:
    for qualified, values in fields.items():
        name = qualified
        target, source = component, live
        if ":" in name:
            namespace, name = name.split(":", 1)
            if not hasattr(target, namespace):
                setattr(target, namespace, Model.AttributeNamespace(namespace))
            target, source = getattr(target, namespace), getattr(source, namespace)
        array = getattr(target, name, None)
        original = getattr(source, name)
        if array is None or array.shape != original.shape:
            array = wp.empty_like(original)
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


@dataclass
class HistoryFrame:
    """Host-side display snapshot, not a resumable solver checkpoint.

    Component dictionaries contain copied NumPy arrays keyed by attribute name
    (``namespace:name`` for custom attributes). Contact and observable arrays
    retain active rows only; solver scratch data and internal state are excluded.
    """

    time: float
    """Simulation time in seconds."""
    state: dict[str, np.ndarray]
    contacts: dict[str, np.ndarray]
    size: int
    """Bytes occupied by copied array data, excluding Python container overhead."""
    touching: int = 0
    depth_mm: float = 0
    body_speed: float = float("nan")
    particle_speed: float = float("nan")
    soft_shape_contacts: int = 0
    soft_self_contacts: int = 0
    observables: dict[str, np.ndarray] = field(default_factory=dict)
    particle_states: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    control: dict[str, np.ndarray] = field(default_factory=dict)
    contact_state: dict[str, np.ndarray] = field(default_factory=dict)
    """Collision input poses and velocities when distinct from the displayed State."""

    @property
    def soft_touching(self) -> int:
        return self.soft_shape_contacts + self.soft_self_contacts


class WorkspaceHistory:
    """Record bounded example display snapshots without rewinding the solver.

    Capture copies declared arrays to the host at rendered frame boundaries.
    Replay uploads a selected snapshot into separate display buffers and invokes
    the example's render callback. The live simulation remains at the newest
    frame and can step again only after playback reaches that frame.

    This adapter relies on conventional example attributes, not arbitrary solver
    checkpoints. Memory and frame limits evict the oldest snapshots; ``index``
    is relative to retained frames and ``first_frame`` counts evicted frames.
    """

    contact_fields = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1")
    soft_contact_fields = ("indices", "particle", "barycentric", "shape", "body_pos", "body_vel", "normal")

    def __init__(self, example: ViewerExample) -> None:
        self.example = example
        # Examples may keep their model inside a simulation object. The viewer
        # owns the model for the displayed layer.
        self.model = example.viewer.model
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
        self.display_state = None
        self.display_contact_state = None
        self.display_control = None
        self.display_particle_states: dict[str, State] = {}
        self.display_contacts = None
        self.display_observables = None
        self._observables_source = None
        self.loaded_index = -1
        self._self_contact_counts = None
        self.dt = float(getattr(example, "frame_dt", 1 / 60))

    def _soft_self_contact_count(self, contacts: Contacts) -> int:
        """Count stored self-contact pairs without copying collision buffers to the CPU."""
        info = getattr(contacts, "soft_self_contact_data", None)
        solver = getattr(self.example, "solver", None)
        if info is None and getattr(solver, "particle_enable_self_contact", False):
            info = getattr(getattr(solver, "_pipeline_contacts", None), "soft_self_contact_data", None)
            if info is None:
                info = getattr(getattr(solver, "trimesh_collision_detector", None), "collision_info", None)
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

    @property
    def replaying(self) -> bool:
        return bool(self.frames) and self.index < len(self.frames) - 1

    def select(self, index: int) -> None:
        self.index = max(0, min(int(index), len(self.frames) - 1))

    def advance(self, steps: float = 1) -> None:
        self._playback_fraction += steps * self.playback_speed
        steps = int(self._playback_fraction)
        self._playback_fraction -= steps
        self.select(self.index + steps)

    @staticmethod
    def _observable_ranges(
        observables: SolverBase.Observables,
        name: str,
        array: wp.array[Any],
        rigid_capacity: int,
        count: int,
        soft_count: int,
    ) -> tuple[tuple[int, int], ...]:
        """Retain active contact rows while preserving each observable's row domain."""
        frequency = observables.get_attribute_frequency(name)
        if frequency == Model.AttributeFrequency.CONTACT:
            return ((0, count), (rigid_capacity, soft_count))
        if frequency == Model.AttributeFrequency.CONTACT_RIGID:
            return ((0, count),)
        if frequency == Model.AttributeFrequency.CONTACT_SOFT:
            return ((0, soft_count),)
        return ((0, len(array)),)

    @staticmethod
    def _snapshot_observable(array: wp.array[Any], ranges: Sequence[tuple[int, int]]) -> np.ndarray:
        parts = [array[start : start + n].numpy() for start, n in ranges if n]
        # Warp rejects a zero-length slice starting at the array's end.
        # Preserve an empty array's dtype/shape without copying inactive rows.
        if not parts:
            return array[:0].numpy() if len(array) else array.numpy()
        return np.concatenate(parts)

    def capture(self, *, replace: bool = False) -> None:
        """Copy the current example frame, evicting old snapshots to stay bounded.

        Args:
            replace: Permit replacing the newest snapshot at the same simulation
                time, for example after an edit while paused. Otherwise repeated
                renders at that time do not trigger another capture.

        Oversized snapshots stop recording rather than transferring unbounded
        device data. A successful capture moves the cursor to the newest frame.
        """
        if not self.recording or self._recording_blocked:
            return
        example = self.example
        time = float(getattr(example, "sim_time", 0.0))
        replace_last = bool(self.frames) and time == self.frames[-1].time
        if replace_last and not replace:
            return
        state = getattr(example, "state_0", getattr(example, "state", None))
        if state is None:
            self.message = "This example does not expose a recordable state."
            self._recording_blocked = True
            return
        arrays = _component_arrays(state, model=self.model)
        control_arrays = _component_arrays(getattr(example, "control", None), model=self.model)
        contact_state = getattr(example, "contact_state", None)
        contact_state_arrays = {
            name: array
            for name in ("body_q", "body_qd")
            if contact_state is not state and (array := getattr(contact_state, name, None)) is not None
        }
        contact_state_bytes = sum(a.capacity for a in contact_state_arrays.values())
        # Coupled examples can draw particles from a separate State. Retain
        # those display arrays too, without copying solver grids or buffers.
        particle_states = {}
        seen = {id(state.particle_q)}
        for name, value in vars(example).items():
            if name in ("state", "state_0", "state_1", "contact_state") or not isinstance(value, State):
                continue
            if value.particle_q is None or id(value.particle_q) in seen:
                continue
            seen.add(id(value.particle_q))
            particle_states[name] = {
                field: getattr(value, field)
                for field in ("particle_q", "particle_qd", "particle_f")
                if getattr(value, field, None) is not None
            }
        particle_bytes = sum(a.capacity for fields in particle_states.values() for a in fields.values())
        # Reject oversized snapshots before any device-to-host transfer.
        control_bytes = sum(a.capacity for a in control_arrays.values())
        if (
            sum(a.capacity for a in arrays.values()) + particle_bytes + control_bytes + contact_state_bytes
            > self.memory_limit // 2
        ):
            self.message = "A state exceeds the 64 MiB snapshot limit. Recording stopped."
            self._recording_blocked = True
            return
        contacts = getattr(example, "contacts", None)
        contact_arrays = {}
        count = soft_count = 0
        if contacts is not None:
            counters = contacts.contact_counters.numpy()
            count = min(int(counters[0]), contacts.rigid_contact_max)
            soft_count = min(int(counters[1]), contacts.soft_contact_max)
            for prefix, fields, n in (
                ("rigid_contact_", self.contact_fields, count),
                ("soft_contact_", self.soft_contact_fields, soft_count),
            ):
                if n:
                    for suffix in fields:
                        name = prefix + suffix
                        array = getattr(contacts, name, None)
                        if array is not None:
                            contact_arrays[name] = (array, n)
        observables = getattr(example, "solver_observables", None)
        observable_arrays = {}
        if observables is not None:
            for name, array in vars(observables).items():
                if not name.startswith("_") and isinstance(array, wp.array):
                    ranges = self._observable_ranges(
                        observables, name, array, contacts.rigid_contact_max if contacts else 0, count, soft_count
                    )
                    observable_arrays[name] = (array, ranges)
        size = (
            sum(a.capacity for a in arrays.values())
            + particle_bytes
            + control_bytes
            + contact_state_bytes
            + sum(n * array.strides[0] for array, n in contact_arrays.values())
            + sum(sum(n for _, n in ranges) * array.strides[0] for array, ranges in observable_arrays.values())
        )
        if size > self.memory_limit // 2:
            self.message = "State and contacts exceed the 64 MiB snapshot limit. Recording stopped."
            self._recording_blocked = True
            return
        saved_state = {name: array.numpy().copy() for name, array in arrays.items()}
        saved_control = {name: array.numpy().copy() for name, array in control_arrays.items()}
        saved_contact_state = {name: array.numpy().copy() for name, array in contact_state_arrays.items()}
        saved_particle_states = {
            name: {field: array.numpy().copy() for field, array in fields.items()}
            for name, fields in particle_states.items()
        }
        saved_contacts = {name: array[:n].numpy().copy() for name, (array, n) in contact_arrays.items()}
        saved_observables = {
            name: self._snapshot_observable(array, ranges) for name, (array, ranges) in observable_arrays.items()
        }
        size = sum(a.nbytes for a in (*saved_state.values(), *saved_contacts.values(), *saved_observables.values()))
        size += sum(a.nbytes for fields in saved_particle_states.values() for a in fields.values())
        size += sum(a.nbytes for a in saved_control.values())
        size += sum(a.nbytes for a in saved_contact_state.values())
        frame = HistoryFrame(
            time,
            saved_state,
            saved_contacts,
            size,
            count,
            body_speed=self.mean_speed(saved_state.get("body_qd")),
            particle_speed=self.mean_speed(saved_state.get("particle_qd")),
            soft_shape_contacts=soft_count,
            soft_self_contacts=self._soft_self_contact_count(contacts) if self.model.particle_count else 0,
            observables=saved_observables,
            particle_states=saved_particle_states,
            control=saved_control,
            contact_state=saved_contact_state,
        )
        if replace_last:
            self.bytes_used -= self.frames.pop().size
        self.frames.append(frame)
        self.record_plot_frame(frame)
        self.bytes_used += size
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
        self.loaded_index = -1

    def render(self) -> None:
        """Render live data or temporarily substitute the selected historical frame.

        Replay buffers are uploaded only when the cursor or their storage changes.
        Conventional example state/contact/time references are restored in a
        finally block, including when rendering fails. The callback must only
        render: these snapshots omit solver state required to resume stepping.
        """
        example = self.example
        if not self.replaying:
            example.render()
            return
        frame = self.frames[self.index]
        if self.display_state is None:
            self.display_state = self.model.state()
        if frame.control and self.display_control is None:
            self.display_control = self.model.control(clone_variables=True)
        count = frame.touching
        soft_count = frame.soft_shape_contacts
        observables = getattr(example, "solver_observables", None)
        live_contacts = getattr(example, "contacts", None)
        # Solver-allocated observables enforce both storage identity and
        # capacity. Give replay its own normally allocated container and bind
        # it to separate contacts; never rebind or overwrite live observables.
        if observables is not self._observables_source:
            self.display_observables = None
            self._observables_source = observables
            self.display_contacts = None
            self.loaded_index = -1
        rigid_capacity, soft_capacity = (
            (live_contacts.rigid_contact_max, live_contacts.soft_contact_max)
            if observables is not None and live_contacts is not None
            else (max(1, count), soft_count)
        )
        if (
            self.display_contacts is None
            or self.display_contacts.rigid_contact_max < rigid_capacity
            or self.display_contacts.soft_contact_max < soft_capacity
        ):
            self.display_contacts = Contacts(rigid_capacity, soft_capacity, device=self.model.device)
            self.display_observables = None
            self.loaded_index = -1
        if observables is not None and self.display_observables is None:
            self.display_observables = example.solver.observables(
                kinds=observables.kinds, contacts=self.display_contacts, requires_grad=False
            )
        if self.loaded_index != self.index:
            live_state = getattr(example, "state_0", getattr(example, "state", None))
            _restore_component(self.display_state, live_state, frame.state)
            if frame.contact_state:
                # Contact witnesses use the collision input frame, which can
                # precede solver corrections to the displayed body poses.
                if self.display_contact_state is None:
                    self.display_contact_state = State()
                _restore_component(self.display_contact_state, example.contact_state, frame.contact_state)
            if frame.control:
                _restore_component(self.display_control, example.control, frame.control)
            for name, fields in frame.particle_states.items():
                if name not in self.display_particle_states:
                    self.display_particle_states[name] = State()
                display = self.display_particle_states[name]
                for field, values in fields.items():
                    array = getattr(display, field, None)
                    live = getattr(getattr(example, name), field)
                    if array is None or array.shape != live.shape:
                        array = wp.empty_like(live)
                        setattr(display, field, array)
                    array.assign(values)
            self.display_contacts.rigid_contact_count.assign(np.array([count], dtype=np.int32))
            self.display_contacts.soft_contact_count.assign(np.array([soft_count], dtype=np.int32))
            for name, values in frame.contacts.items():
                getattr(self.display_contacts, name)[: len(values)].assign(values)
            for name, values in frame.observables.items():
                array = getattr(self.display_observables, name)
                ranges = self._observable_ranges(
                    self.display_observables, name, array, self.display_contacts.rigid_contact_max, count, soft_count
                )
                offset = 0
                for start, n in ranges:
                    if n:
                        array[start : start + n].assign(values[offset : offset + n])
                    offset += n
            self.loaded_index = self.index
        saved = {}
        display = {
            "state_0": self.display_state,
            "state_1": self.display_state,
            "state": self.display_state,
            "contact_state": self.display_contact_state if frame.contact_state else self.display_state,
            "contacts": self.display_contacts,
            "solver_observables": self.display_observables,
            "sim_time": frame.time,
        }
        display.update({name: self.display_particle_states[name] for name in frame.particle_states})
        try:
            for name, value in display.items():
                if hasattr(example, name):
                    saved[name] = getattr(example, name)
                    setattr(example, name, value)
            example.render()
        finally:
            for name, value in saved.items():
                setattr(example, name, value)

    @staticmethod
    def mean_speed(velocity: ArrayLike | None) -> float:
        """Mean linear speed, using velocities already copied for history."""
        if velocity is None or len(velocity) == 0:
            return float("nan")
        linear = np.asarray(velocity)[:, :3]
        return float(np.sqrt(np.einsum("ij,ij->i", linear, linear)).mean(dtype=np.float64))

    def record_plot_frame(self, frame: HistoryFrame) -> None:
        # A fixed ring keeps plot work independent of retained state size.
        if not hasattr(self, "_plot_values"):
            self._plot_values = np.empty((self.max_frames + 64, 5), dtype=np.float64)
        index = (self.first_frame + len(self.frames) - 1) % len(self._plot_values)
        self._plot_values[index] = (
            frame.touching,
            getattr(frame, "soft_shape_contacts", 0),
            getattr(frame, "soft_self_contacts", 0),
            frame.body_speed,
            frame.particle_speed,
        )
        self._plot_revision = getattr(self, "_plot_revision", 0) + 1

    def plot_samples(self, low: int, high: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reduce an inclusive retained-frame range to approximately ``bins`` samples.

        Return bin centers, edges, three contact-count series, and two mean-speed
        series. Contacts use maxima to preserve brief impacts; motion uses means.
        All inputs come from the host-side plot ring, with no device readback.
        """
        key = (self.first_frame, low, high, bins, getattr(self, "_plot_revision", 0))
        if getattr(self, "_plot_cache_key", None) != key:
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
        return min(len(self.frames) - 1, bisect_left(self.frames, time, key=lambda frame: frame.time))

    def set_recording(self, enabled: bool) -> None:
        """Start a fresh bounded recording, or retain the current snapshots."""
        self.recording = enabled
        if enabled:
            self.message = ""
            self._recording_blocked = False
            self.frames.clear()
            self.index = self.bytes_used = self.first_frame = 0
            self.loaded_index = -1
