# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from enum import Enum
from typing import Literal

import numpy as np
import warp as wp

from ..sim import Contacts, Model, State
from ..utils.selection import match_labels

# Object type constants used in the sensing-object transform kernel.
_OBJ_TYPE_TOTAL = 0
_OBJ_TYPE_SHAPE = 1
_OBJ_TYPE_BODY = 2


@wp.kernel(enable_backward=False)
def compute_sensing_obj_transforms_kernel(
    indices: wp.array(dtype=wp.int32),
    obj_types: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    # output
    transforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    idx = indices[tid]
    obj_type = obj_types[tid]
    if obj_type == wp.static(_OBJ_TYPE_BODY):
        transforms[tid] = body_q[idx]
    elif obj_type == wp.static(_OBJ_TYPE_SHAPE):
        body_idx = shape_body[idx]
        if body_idx >= 0:
            transforms[tid] = wp.transform_multiply(body_q[body_idx], shape_transform[idx])
        else:
            transforms[tid] = shape_transform[idx]


@wp.kernel(enable_backward=False)
def accumulate_contact_forces_kernel(
    num_contacts: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_force: wp.array(dtype=wp.spatial_vector),
    sensing_shape_to_row: wp.array(dtype=wp.int32),
    counterpart_shape_to_col: wp.array(dtype=wp.int32),
    # output
    force_matrix: wp.array2d(dtype=wp.vec3),
    total_force: wp.array(dtype=wp.vec3),
):
    """Accumulate per-contact forces into total and/or per-counterpart arrays. Parallelizes over contacts."""
    con_idx = wp.tid()
    if con_idx >= num_contacts[0]:
        return

    shape0 = contact_shape0[con_idx]
    shape1 = contact_shape1[con_idx]
    assert shape0 >= 0 and shape1 >= 0
    force = wp.spatial_top(contact_force[con_idx])

    row0 = sensing_shape_to_row[shape0]
    row1 = sensing_shape_to_row[shape1]

    # total force
    if total_force:
        if row0 >= 0:
            wp.atomic_add(total_force, row0, force)
        if row1 >= 0:
            wp.atomic_add(total_force, row1, -force)

    # per-counterpart forces
    if force_matrix:
        col0 = counterpart_shape_to_col[shape0]
        col1 = counterpart_shape_to_col[shape1]
        if row0 >= 0 and col1 >= 0:
            wp.atomic_add(force_matrix, row0, col1, force)
        if row1 >= 0 and col0 >= 0:
            wp.atomic_add(force_matrix, row1, col0, -force)


@wp.kernel(enable_backward=False)
def expand_body_to_shape_kernel(
    body_to_row: wp.array(dtype=wp.int32),
    body_to_col: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    # output
    shape_to_row: wp.array(dtype=wp.int32),
    shape_to_col: wp.array(dtype=wp.int32),
):
    """Expand body-indexed maps to shape-indexed arrays. Parallelizes over shapes."""
    tid = wp.tid()
    body = shape_body[tid]

    if body_to_row:
        row = -1
        if body >= 0:
            row = body_to_row[body]
        shape_to_row[tid] = row

    if body_to_col:
        col = -1
        if body >= 0:
            col = body_to_col[body]
        shape_to_col[tid] = col


def _check_index_bounds(indices: list[int], count: int, param_name: str, entity_name: str) -> None:
    """Raise IndexError if any index is out of range [0, count)."""
    for idx in indices:
        if idx < 0 or idx >= count:
            raise IndexError(f"{param_name} contains index {idx}, but model only has {count} {entity_name}")


def _split_globals(indices: list[int], local_start: int, tail_global_start: int):
    """Partition sorted shape/body indices into (globals, locals) based on world boundaries."""
    head = 0
    while head < len(indices) and indices[head] < local_start:
        head += 1
    tail = len(indices)
    while tail > head and indices[tail - 1] >= tail_global_start:
        tail -= 1
    return indices[:head] + indices[tail:], indices[head:tail]


def _normalize_world_start(ws: list[int], world_count: int) -> list[int]:
    """Remap all-global entities into one implicit world when no ``add_world()`` calls were made."""
    n = ws[-1]  # total entity count
    has_no_local_entities = ws[0] == ws[-2]
    if has_no_local_entities:
        assert world_count <= 1, (
            f"No local entities but world_count={world_count}"
        )  # internal invariant from ModelBuilder
        return [0, n, n]
    return ws


def _ensure_sorted_unique(indices: list[int], param_name: str) -> list[int]:
    """Return *indices* in strictly increasing order; duplicates are not allowed.

    Raises:
        ValueError: If *indices* contains duplicate values.
    """
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1]:
            raise ValueError(f"{param_name} contains duplicate index {indices[i]}")
        if indices[i] < indices[i - 1]:
            return _ensure_sorted_unique(sorted(indices), param_name)
    return indices


def _assign_counterpart_columns(
    c_globals: list[int],
    c_locals: list[int],
    counterpart_world_start: list[int],
    world_count: int,
    n_entities: int,
) -> tuple[np.ndarray, int, list[list[int]]]:
    """Build counterpart-to-column mapping and per-world counterpart lists.

    Returns:
        col_map: Array mapping each entity index to its column, or -1 if not a counterpart.
        max_cols: Maximum column count across all worlds.
        counterparts_by_world: Per-world list of counterpart indices (globals + locals).
    """
    col_map = np.full(n_entities, -1, dtype=np.int32)

    for col, idx in enumerate(c_globals):
        col_map[idx] = col
    n_global_cols = len(c_globals)

    counterparts_by_world: list[list[int]] = []
    max_cols = n_global_cols
    n_locals = len(c_locals)
    i = 0  # cursor into c_locals
    for w in range(world_count):
        local_col = n_global_cols
        cur_world_locals: list[int] = []
        world_end = counterpart_world_start[w + 1]
        while i < n_locals and c_locals[i] < world_end:
            col_map[c_locals[i]] = local_col
            cur_world_locals.append(c_locals[i])
            local_col += 1
            i += 1
        max_cols = max(max_cols, local_col)
        counterparts_by_world.append(c_globals + cur_world_locals)
    return col_map, max_cols, counterparts_by_world


class SensorContact:
    """Measures contact forces on a set of **sensing objects** (bodies or shapes).

    In its simplest form the sensor reports :attr:`total_force` — the total contact force on each sensing object.
    Optionally, specify **counterparts** to get a per-counterpart breakdown in :attr:`force_matrix`.

    :attr:`total_force` and :attr:`force_matrix` are each nullable: ``total_force`` is ``None`` when
    ``measure_total=False``; ``force_matrix`` is ``None`` when no counterparts are specified.

    .. rubric:: Multi-world behavior

    When the model contains multiple worlds, counterpart mappings are resolved per-world. The collision pipeline and
    solver are expected to produce only within-world contacts, so cross-world force accumulation does not arise in
    practice. Global counterparts (e.g. ground plane) contribute to every world they contact.

    In single-world models where no ``add_world()`` call was made (all entities are global / ``world=-1``), the sensor
    treats the entire model as one implicit world and all entities are valid sensing objects.

    When counterparts are specified, the force matrix has shape ``(sum_of_sensors_across_worlds, max_counterparts)``
    where ``max_counterparts`` is the maximum counterpart count of any single world. Row order matches
    :attr:`sensing_obj_idx`. Columns beyond a world's own counterpart count are zero-padded.

    :attr:`sensing_obj_idx` and :attr:`counterpart_indices` are flat lists that describe the structure of the output
    arrays.

    .. rubric:: Terms

    - **Sensing object** -- body or shape carrying a contact sensor.
    - **Counterpart** -- the other body or shape in a contact interaction.

    .. rubric:: Construction and update order

    ``SensorContact`` requests the ``force`` extended attribute from the model at init, so a :class:`~newton.Contacts`
    object created afterwards (via :meth:`Model.contacts() <newton.Model.contacts>` or directly) will include it
    automatically.

    :meth:`update` reads from ``contacts.force``. Call ``solver.update_contacts(contacts)`` before
    ``sensor.update()`` so that contact forces are current.

    Parameters that select bodies or shapes accept label patterns -- see :ref:`label-matching`.

    Example:
        Measure total contact force on a sphere resting on the ground:

        .. testcode::

            import warp as wp
            import newton
            from newton.sensors import SensorContact

            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform((0, 0, 0.1), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.1, label="ball")
            model = builder.finalize()

            sensor = SensorContact(model, sensing_obj_shapes="ball")
            solver = newton.solvers.SolverMuJoCo(model)
            state = model.state()
            contacts = model.contacts()

            solver.step(state, state, None, None, dt=1.0 / 60.0)
            solver.update_contacts(contacts)
            sensor.update(state, contacts)
            force = sensor.total_force.numpy()  # (n_sensing_objs, 3)

    Raises:
        ValueError: If the configuration of sensing/counterpart objects is invalid.
    """

    class ObjectType(Enum):
        """Deprecated. Type tag for entries in legacy :attr:`sensing_objs` and :attr:`counterparts` properties."""

        TOTAL = _OBJ_TYPE_TOTAL
        """Total force entry."""

        SHAPE = _OBJ_TYPE_SHAPE
        """Individual shape."""

        BODY = _OBJ_TYPE_BODY
        """Individual body."""

    sensing_obj_type: Literal["body", "shape"]
    """Whether :attr:`sensing_obj_idx` contains body indices (``"body"``) or shape indices (``"shape"``)."""

    sensing_obj_idx: list[int]
    """Body or shape index per sensing object, matching the row of output arrays. For ``list[int]`` inputs the caller's
    order is preserved; for string patterns the order follows ascending body/shape index."""

    counterpart_type: Literal["body", "shape"] | None
    """Whether :attr:`counterpart_indices` contains body indices (``"body"``) or shape indices (``"shape"``).
    ``None`` when no counterparts are specified."""

    counterpart_indices: list[list[int]]
    """Counterpart body or shape indices per sensing object. ``counterpart_indices[i]`` lists the counterparts for row ``i``. Global counterparts appear first, followed by per-world locals in ascending index order."""

    total_force: wp.array(dtype=wp.vec3) | None
    """Total contact force [N] per sensing object, shape ``(n_sensing_objs,)``, dtype :class:`vec3`.
    ``None`` when ``measure_total=False``."""

    force_matrix: wp.array2d(dtype=wp.vec3) | None
    """Per-counterpart contact forces [N], shape ``(n_sensing_objs, max_counterparts)``, dtype :class:`vec3`.
    Entry ``[i, j]`` is the force on sensing object ``i`` from counterpart ``counterpart_indices[i][j]``, in world
    frame. ``None`` when no counterparts are specified."""

    sensing_obj_transforms: wp.array(dtype=wp.transform)
    """World-frame transforms of sensing objects [m, unitless quaternion],
    shape ``(n_sensing_objs,)``, dtype :class:`transform`."""

    def __init__(
        self,
        model: Model,
        *,
        sensing_obj_bodies: str | list[str] | list[int] | None = None,
        sensing_obj_shapes: str | list[str] | list[int] | None = None,
        counterpart_bodies: str | list[str] | list[int] | None = None,
        counterpart_shapes: str | list[str] | list[int] | None = None,
        measure_total: bool = True,
        verbose: bool | None = None,
        request_contact_attributes: bool = True,
        # deprecated
        include_total: bool | None = None,
    ):
        """Initialize the SensorContact.

        Exactly one of ``sensing_obj_bodies`` or ``sensing_obj_shapes`` must be specified to define the sensing
        objects. At most one of ``counterpart_bodies`` or ``counterpart_shapes`` may be specified. If neither is
        specified, only :attr:`total_force` is available (no per-counterpart breakdown).

        Args:
            model: The simulation model providing shape/body definitions and world layout.
            sensing_obj_bodies: List of body indices, single pattern to match
                against body labels, or list of patterns where any one matches.
            sensing_obj_shapes: List of shape indices, single pattern to match
                against shape labels, or list of patterns where any one matches.
            counterpart_bodies: List of body indices, single pattern to match
                against body labels, or list of patterns where any one matches.
            counterpart_shapes: List of shape indices, single pattern to match
                against shape labels, or list of patterns where any one matches.
            measure_total: If True (default), :attr:`total_force` is allocated. If False, :attr:`total_force` is None.
            verbose: If True, print details. If None, uses ``wp.config.verbose``.
            request_contact_attributes: If True (default), transparently request the extended contact attribute
                ``force`` from the model.
            include_total: Deprecated. Use ``measure_total`` instead.
        """
        if include_total is not None:
            warnings.warn(
                "SensorContact: 'include_total' is deprecated, use 'measure_total' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            measure_total = include_total

        if (sensing_obj_bodies is None) == (sensing_obj_shapes is None):
            raise ValueError("Exactly one of `sensing_obj_bodies` and `sensing_obj_shapes` must be specified")

        if (counterpart_bodies is not None) and (counterpart_shapes is not None):
            raise ValueError("At most one of `counterpart_bodies` and `counterpart_shapes` may be specified.")

        self.device = model.device
        self.verbose = verbose if verbose is not None else wp.config.verbose

        # request contact force attribute
        if request_contact_attributes:
            model.request_contact_attributes("force")

        if sensing_obj_bodies is not None:
            s_bodies = match_labels(model.body_label, sensing_obj_bodies)
            _check_index_bounds(s_bodies, len(model.body_label), "sensing_obj_bodies", "bodies")
            s_shapes = []
        else:
            s_bodies = []
            s_shapes = match_labels(model.shape_label, sensing_obj_shapes)
            _check_index_bounds(s_shapes, len(model.shape_label), "sensing_obj_shapes", "shapes")

        using_counterparts = True
        if counterpart_bodies is not None:
            c_bodies = match_labels(model.body_label, counterpart_bodies)
            _check_index_bounds(c_bodies, len(model.body_label), "counterpart_bodies", "bodies")
            c_shapes = []
        elif counterpart_shapes is not None:
            c_bodies = []
            c_shapes = match_labels(model.shape_label, counterpart_shapes)
            _check_index_bounds(c_shapes, len(model.shape_label), "counterpart_shapes", "shapes")
        else:
            c_shapes = []
            c_bodies = []
            using_counterparts = False

        world_count = model.world_count

        # Determine whether sensing objects and counterparts are body-level or shape-level.
        sensing_is_body = sensing_obj_bodies is not None
        counterpart_is_body = counterpart_bodies is not None
        sensing_indices = s_bodies if sensing_is_body else s_shapes
        counterpart_indices = c_bodies if counterpart_is_body else c_shapes

        sensing_world_start = _normalize_world_start(
            (model.body_world_start if sensing_is_body else model.shape_world_start).list(), world_count
        )
        counterpart_world_start = _normalize_world_start(
            (model.body_world_start if counterpart_is_body else model.shape_world_start).list(), world_count
        )

        sensing_indices_ordered = list(sensing_indices)  # preserve user's original order
        sensing_indices = _ensure_sorted_unique(
            sensing_indices, "sensing_obj_bodies" if sensing_is_body else "sensing_obj_shapes"
        )
        counterpart_indices = _ensure_sorted_unique(
            counterpart_indices, "counterpart_bodies" if counterpart_is_body else "counterpart_shapes"
        )

        if not sensing_indices:
            raise ValueError(
                f"No {'bodies' if sensing_is_body else 'shapes'} matched the sensing object pattern(s). "
                "Check that the labels exist in the model."
            )

        if using_counterparts and not counterpart_indices:
            raise ValueError(
                f"No {'bodies' if counterpart_is_body else 'shapes'} matched the counterpart pattern(s). "
                "Check that the labels exist in the model."
            )

        s_globals, _ = _split_globals(sensing_indices, sensing_world_start[0], sensing_world_start[world_count])
        if s_globals:
            raise ValueError(f"Global bodies/shapes (world=-1) cannot be sensing objects. Global indices: {s_globals}")

        # Assign rows to sensing objects
        n_entities_s = len(model.body_label) if sensing_is_body else model.shape_count
        sensing_to_row = np.full(n_entities_s, -1, dtype=np.int32)
        sensing_to_row[sensing_indices_ordered] = np.arange(len(sensing_indices_ordered), dtype=np.int32)

        # Assign columns to counterparts: first global, then local
        c_globals, c_locals = _split_globals(
            counterpart_indices, counterpart_world_start[0], counterpart_world_start[world_count]
        )
        n_entities_c = len(model.body_label) if counterpart_is_body else model.shape_count
        counterpart_to_col, max_readings, counterparts_by_world = _assign_counterpart_columns(
            c_globals, c_locals, counterpart_world_start, world_count, n_entities_c
        )

        if not measure_total and max_readings == 0:
            raise ValueError(
                "Sensor configured with measure_total=False and no counterparts — "
                "at least one output (total_force or force_matrix) must be enabled."
            )

        n_rows = len(sensing_indices)

        # --- Build Warp arrays ---
        n_shapes = model.shape_count
        body_to_row = None
        body_to_col = None

        if sensing_is_body:
            body_to_row = wp.array(sensing_to_row, dtype=wp.int32, device=self.device)
            self._sensing_shape_to_row = wp.full(n_shapes, -1, dtype=wp.int32, device=self.device)
        else:
            self._sensing_shape_to_row = wp.array(sensing_to_row, dtype=wp.int32, device=self.device)

        if counterpart_is_body:
            body_to_col = wp.array(counterpart_to_col, dtype=wp.int32, device=self.device)
            self._counterpart_shape_to_col = wp.full(n_shapes, -1, dtype=wp.int32, device=self.device)
        else:
            self._counterpart_shape_to_col = wp.array(counterpart_to_col, dtype=wp.int32, device=self.device)

        if sensing_is_body or counterpart_is_body:
            wp.launch(
                expand_body_to_shape_kernel,
                dim=n_shapes,
                inputs=[
                    body_to_row if sensing_is_body else None,
                    body_to_col if counterpart_is_body else None,
                    model.shape_body,
                ],
                outputs=[
                    self._sensing_shape_to_row,
                    self._counterpart_shape_to_col,
                ],
                device=self.device,
            )

        total_cols = int(measure_total) + max_readings
        self._net_force = wp.zeros((n_rows, total_cols), dtype=wp.vec3, device=self.device)

        if measure_total:
            self.total_force = self._net_force[:, 0]
        else:
            self.total_force = None

        if max_readings > 0:
            self.force_matrix = self._net_force[:, int(measure_total) :]
        else:
            self.force_matrix = None

        self.sensing_obj_type = "body" if sensing_is_body else "shape"
        self.counterpart_type = "body" if counterpart_is_body else ("shape" if counterpart_indices else None)
        self.sensing_obj_idx = sensing_indices_ordered

        # Map each sensing object to its world's counterpart list.
        world_starts = np.array(sensing_world_start[:world_count])
        worlds = np.searchsorted(world_starts, sensing_indices_ordered, side="right") - 1
        self.counterpart_indices = [counterparts_by_world[w] for w in worlds]

        if self.verbose:
            print("SensorContact initialized:")
            print(f"  Sensing objects: {n_rows} ({self.sensing_obj_type}s)")
            print(
                f"  Counterpart columns: {max_readings}"
                + (f" ({self.counterpart_type}s)" if self.counterpart_type else "")
            )
            print(
                f"  total_force: {'yes' if measure_total else 'no'}, "
                f"force_matrix: {'yes' if max_readings > 0 else 'no'}"
            )

        self._model = model
        self._sensing_obj_indices = wp.array(sensing_indices_ordered, dtype=wp.int32, device=self.device)
        obj_type = _OBJ_TYPE_BODY if sensing_is_body else _OBJ_TYPE_SHAPE
        self._sensing_obj_types = wp.full(n_rows, obj_type, dtype=wp.int32, device=self.device)
        self.sensing_obj_transforms = wp.zeros(n_rows, dtype=wp.transform, device=self.device)

        self._init_deprecated_shims(measure_total, world_count, worlds)

    def _init_deprecated_shims(self, measure_total: bool, world_count: int, sensing_obj_worlds: np.ndarray):
        """Store data needed by deprecated backward-compatible properties.

        The properties themselves are computed lazily on first access and cached.
        """
        self._measure_total = measure_total
        self._world_count = world_count
        self._sensing_obj_worlds = sensing_obj_worlds

    @property
    def shape(self) -> tuple[int, int]:
        """Deprecated. Dimensions of :attr:`net_force`."""
        warnings.warn(
            "SensorContact.shape is deprecated. Use total_force.shape / force_matrix.shape instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return tuple(self._net_force.shape)

    @property
    def sensing_objs(self) -> list[list[tuple[int, ObjectType]]]:
        """Deprecated. Use :attr:`sensing_obj_idx` and :attr:`sensing_obj_type` instead."""
        warnings.warn(
            "SensorContact.sensing_objs is deprecated. Use 'sensing_obj_idx' and 'sensing_obj_type' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_deprecated_sensing_objs"):
            return self._deprecated_sensing_objs
        obj_type = self.ObjectType.BODY if self.sensing_obj_type == "body" else self.ObjectType.SHAPE
        result: list[list[tuple[int, SensorContact.ObjectType]]] = [[] for _ in range(self._world_count)]
        for i, idx in enumerate(self.sensing_obj_idx):
            result[int(self._sensing_obj_worlds[i])].append((idx, obj_type))
        self._deprecated_sensing_objs = result
        return result

    @property
    def counterparts(self) -> list[list[tuple[int, ObjectType]]]:
        """Deprecated. Use :attr:`counterpart_indices` and :attr:`counterpart_type` instead."""
        warnings.warn(
            "SensorContact.counterparts is deprecated. Use 'counterpart_indices' and 'counterpart_type' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_deprecated_counterparts"):
            return self._deprecated_counterparts
        cp_type = (
            self.ObjectType.BODY
            if self.counterpart_type == "body"
            else self.ObjectType.SHAPE
            if self.counterpart_type == "shape"
            else None
        )
        result: list[list[tuple[int, SensorContact.ObjectType]]] = [[] for _ in range(self._world_count)]
        seen_worlds: set[int] = set()
        for i in range(len(self.sensing_obj_idx)):
            w = int(self._sensing_obj_worlds[i])
            if w in seen_worlds:
                continue
            seen_worlds.add(w)
            entries: list[tuple[int, SensorContact.ObjectType]] = []
            if self._measure_total:
                entries.append((-1, self.ObjectType.TOTAL))
            if cp_type is not None:
                for idx in self.counterpart_indices[i]:
                    entries.append((idx, cp_type))
            result[w] = entries
        self._deprecated_counterparts = result
        return result

    @property
    def reading_indices(self) -> list[list[list[int]]]:
        """Deprecated. Active counterpart indices per sensing object, per world."""
        warnings.warn(
            "SensorContact.reading_indices is deprecated. Use 'counterpart_indices' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_deprecated_reading_indices"):
            return self._deprecated_reading_indices
        result: list[list[list[int]]] = [[] for _ in range(self._world_count)]
        for i in range(len(self.sensing_obj_idx)):
            w = int(self._sensing_obj_worlds[i])
            n_active = int(self._measure_total) + len(self.counterpart_indices[i])
            result[w].append(list(range(n_active)))
        self._deprecated_reading_indices = result
        return result

    def update(self, state: State | None, contacts: Contacts):
        """Update the contact sensor readings based on the provided state and contacts.

        Computes world-frame transforms for all sensing objects and evaluates contact forces
        (total and/or per-counterpart, depending on sensor configuration).

        Args:
            state: The simulation state providing body transforms, or None to skip
                the transform update.
            contacts: The contact data to evaluate.

        Raises:
            ValueError: If ``contacts.force`` is None.
            ValueError: If ``contacts.device`` does not match the sensor's device.
        """
        # update sensing object transforms
        n = len(self._sensing_obj_indices)
        if n > 0 and state is not None and state.body_q is not None:
            wp.launch(
                compute_sensing_obj_transforms_kernel,
                dim=n,
                inputs=[
                    self._sensing_obj_indices,
                    self._sensing_obj_types,
                    self._model.shape_body,
                    self._model.shape_transform,
                    state.body_q,
                ],
                outputs=[self.sensing_obj_transforms],
                device=self.device,
            )

        if contacts.force is None:
            raise ValueError(
                "SensorContact requires a ``Contacts`` object with ``force`` allocated. "
                "Create ``SensorContact`` before ``Contacts`` for automatically requesting it."
            )
        if contacts.device != self.device:
            raise ValueError(f"Contacts device ({contacts.device}) does not match sensor device ({self.device}).")
        self._eval_forces(contacts)

    @property
    def net_force(self) -> wp.array2d:
        """Deprecated. Use :attr:`total_force` and :attr:`force_matrix` instead."""
        warnings.warn(
            "SensorContact.net_force is deprecated. Use 'total_force' for total forces "
            "and 'force_matrix' for per-counterpart forces.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._net_force

    def _eval_forces(self, contacts: Contacts):
        """Zero and recompute :attr:`total_force` and :attr:`force_matrix` from the given contacts."""
        self._net_force.zero_()
        wp.launch(
            accumulate_contact_forces_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.force,
                self._sensing_shape_to_row,
                self._counterpart_shape_to_col,
            ],
            outputs=[self.force_matrix, self.total_force],
            device=self.device,
        )
