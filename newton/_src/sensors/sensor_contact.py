# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import warp as wp

from ..sim import Contacts, Model, State
from ..sim.contacts import contact_surface_point
from ..utils.selection import match_labels

_UNSET = object()

_SENSING_KIND_SHAPE = 1
_SENSING_KIND_BODY = 2

_SENSING_OBJ_IDX_DEPRECATION_MSG = (
    "SensorContact.sensing_obj_idx is deprecated; use SensorContact.sensing_indices. "
    "The alias will be removed in a future release."
)
_SENSING_OBJ_TYPE_DEPRECATION_MSG = (
    "SensorContact.sensing_obj_type is deprecated; use SensorContact.sensing_type. "
    "The alias will be removed in a future release."
)
_SENSING_OBJ_TRANSFORMS_DEPRECATION_MSG = (
    "SensorContact.sensing_obj_transforms is deprecated; use SensorContact.sensing_transforms. "
    "The alias will be removed in a future release."
)
_SENSING_OBJ_BODIES_DEPRECATION_MSG = (
    "SensorContact(..., sensing_obj_bodies=...) is deprecated; use sensing_bodies=... instead. "
    "The alias will be removed in a future release."
)
_SENSING_OBJ_SHAPES_DEPRECATION_MSG = (
    "SensorContact(..., sensing_obj_shapes=...) is deprecated; use sensing_shapes=... instead. "
    "The alias will be removed in a future release."
)


@wp.kernel(enable_backward=False)
def compute_sensing_transforms_kernel(
    indices: wp.array[wp.int32],
    sensing_kinds: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    # output
    transforms: wp.array[wp.transform],
):
    tid = wp.tid()
    index = indices[tid]
    sensing_kind = sensing_kinds[tid]
    if sensing_kind == wp.static(_SENSING_KIND_BODY):
        transforms[tid] = body_q[index]
    elif sensing_kind == wp.static(_SENSING_KIND_SHAPE):
        body_index = shape_body[index]
        if body_index >= 0:
            transforms[tid] = wp.transform_multiply(body_q[body_index], shape_transform[index])
        else:
            transforms[tid] = shape_transform[index]


@wp.kernel(enable_backward=False)
def accumulate_contact_forces_kernel(
    num_contacts: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_force: wp.array[wp.spatial_vector],
    contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    sensing_shape_to_row: wp.array[wp.int32],
    counterpart_shape_to_col: wp.array[wp.int32],
    # output
    force_matrix: wp.array2d[wp.vec3],
    total_force: wp.array[wp.vec3],
    force_matrix_friction: wp.array2d[wp.vec3],
    total_force_friction: wp.array[wp.vec3],
    position_matrix: wp.array2d[wp.vec3],
    position_weight: wp.array2d[float],
):
    """Accumulate per-contact forces, friction, and weighted positions. Parallelizes over contacts."""
    contact_index = wp.tid()
    if contact_index >= num_contacts[0]:
        return

    shape0 = contact_shape0[contact_index]
    shape1 = contact_shape1[contact_index]
    assert shape0 >= 0 and shape1 >= 0
    force = wp.spatial_top(contact_force[contact_index])

    # Decompose into normal and friction (tangential) components
    n = contact_normal[contact_index]
    len_sq = wp.dot(n, n)
    if wp.abs(len_sq - 1.0) > 1.0e-4:
        n = wp.normalize(n)
    friction = force - wp.dot(force, n) * n

    row0 = sensing_shape_to_row[shape0]
    row1 = sensing_shape_to_row[shape1]

    # total force and friction
    if total_force:
        assert total_force_friction
        if row0 >= 0:
            wp.atomic_add(total_force, row0, force)
            wp.atomic_add(total_force_friction, row0, friction)
        if row1 >= 0:
            wp.atomic_add(total_force, row1, -force)
            wp.atomic_add(total_force_friction, row1, -friction)

    # per-counterpart forces and friction
    if force_matrix:
        assert force_matrix_friction
        col0 = counterpart_shape_to_col[shape0]
        col1 = counterpart_shape_to_col[shape1]
        matched0 = row0 >= 0 and col1 >= 0
        matched1 = row1 >= 0 and col0 >= 0
        if matched0:
            wp.atomic_add(force_matrix, row0, col1, force)
            wp.atomic_add(force_matrix_friction, row0, col1, friction)
        if matched1:
            wp.atomic_add(force_matrix, row1, col0, -force)
            wp.atomic_add(force_matrix_friction, row1, col0, -friction)

        if position_matrix:
            assert position_weight
            weight = wp.length(force)
            if weight > 0.0 and (matched0 or matched1):
                body0 = shape_body[shape0]
                body1 = shape_body[shape1]
                transform0 = wp.where(body0 >= 0, body_q[wp.max(body0, 0)], wp.transform_identity())
                transform1 = wp.where(body1 >= 0, body_q[wp.max(body1, 0)], wp.transform_identity())
                point0_world = contact_surface_point(
                    transform0, contact_point0[contact_index], contact_offset0[contact_index]
                )
                point1_world = contact_surface_point(
                    transform1, contact_point1[contact_index], contact_offset1[contact_index]
                )
                midpoint = 0.5 * (point0_world + point1_world)
                weighted_midpoint = weight * midpoint

                if matched0:
                    wp.atomic_add(position_matrix, row0, col1, weighted_midpoint)
                    wp.atomic_add(position_weight, row0, col1, weight)
                if matched1:
                    wp.atomic_add(position_matrix, row1, col0, weighted_midpoint)
                    wp.atomic_add(position_weight, row1, col0, weight)


@wp.kernel(enable_backward=False)
def normalize_contact_positions_kernel(
    position_matrix: wp.array2d[wp.vec3],
    position_weight: wp.array2d[float],
):
    """Normalize force-weighted contact position sums in place; entries with no contributing contacts stay zero."""
    row, col = wp.tid()
    weight = position_weight[row, col]
    if weight > 0.0:
        position_matrix[row, col] /= weight


@wp.kernel(enable_backward=False)
def expand_body_to_shape_kernel(
    body_to_row: wp.array[wp.int32],
    body_to_col: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    # output
    shape_to_row: wp.array[wp.int32],
    shape_to_col: wp.array[wp.int32],
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
    for index in indices:
        if index < 0 or index >= count:
            raise IndexError(f"{param_name} contains index {index}, but model only has {count} {entity_name}")


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

    for col, index in enumerate(c_globals):
        col_map[index] = col
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
    """Measures contact forces, friction, and force-weighted positions on **sensing objects** (bodies or shapes).

    In its simplest form the sensor reports :attr:`total_force` — the total contact force on each sensing object — and
    :attr:`total_force_friction`, its tangential (friction) component. Optionally, specify **counterparts** to separate
    these measurements by interacting body or shape. :attr:`force_matrix` reports the per-counterpart contact forces,
    :attr:`force_matrix_friction` reports their frictional (tangential) components, and :attr:`position_matrix` reports
    the average interaction positions. In each matrix, row ``i`` corresponds to ``sensing_indices[i]`` and column ``j``
    within that row corresponds to ``counterpart_indices[i][j]``. Columns beyond a row's counterpart list are zero
    padding.

    Each :attr:`position_matrix` entry is the average of the contact midpoints accumulated at that index, weighted by
    contact-force magnitude. It provides a representative location for where each counterpart interacts with the
    sensing object. Force weighting reduces the influence of weak contacts, so adding or removing a low-force contact
    perturbs the reported position less.

    :attr:`total_force` and :attr:`total_force_friction` are ``None`` when ``measure_total=False``. Per-counterpart
    outputs :attr:`force_matrix`, :attr:`force_matrix_friction`, and :attr:`position_matrix` are ``None`` when
    no counterparts are specified.

    .. rubric:: Multi-world behavior

    When the model contains multiple worlds, counterpart mappings are resolved per-world. The collision pipeline and
    solver are expected to produce only within-world contacts, so cross-world force accumulation does not arise in
    practice. Global counterparts (e.g. ground plane) contribute to every world they contact.

    In single-world models where no ``add_world()`` call was made (all entities are global / ``world=-1``), the sensor
    treats the entire model as one implicit world and all entities are valid sensing objects.

    When counterparts are specified, the per-counterpart matrices have shape
    ``(sum_of_sensors_across_worlds, max_counterparts)``, where ``max_counterparts`` is the maximum counterpart count
    of any single world. Row order matches
    :attr:`sensing_indices`. Columns beyond a world's own counterpart count are zero-padded.

    :attr:`sensing_indices` and :attr:`counterpart_indices` are flat lists that describe the structure of the output
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

            sensor = SensorContact(model, sensing_shapes="ball")
            solver = newton.solvers.SolverMuJoCo(model)
            state = model.state()
            contacts = model.contacts()

            solver.step(state, state, None, None, dt=1.0 / 60.0)
            solver.update_contacts(contacts)
            sensor.update(state, contacts)
            force = sensor.total_force.numpy()  # (n_sensing, 3)

    Raises:
        ValueError: If the configuration of sensing/counterpart objects is invalid.
    """

    sensing_indices: list[int]
    """Body or shape index per sensing object, matching the row of output arrays. For ``list[int]`` inputs the caller's
    order is preserved; for string patterns the order follows ascending body/shape index."""

    sensing_type: Literal["body", "shape"]
    """Whether :attr:`sensing_indices` contains body indices (``"body"``) or shape indices (``"shape"``)."""

    counterpart_indices: list[list[int]]
    """Counterpart body or shape indices per sensing object. ``counterpart_indices[i]`` lists the counterparts for row
    ``i``. Global counterparts appear first, followed by per-world locals in ascending index order."""

    counterpart_type: Literal["body", "shape"] | None
    """Whether :attr:`counterpart_indices` contains body indices (``"body"``) or shape indices (``"shape"``).
    ``None`` when no counterparts are specified."""

    total_force: wp.array[wp.vec3] | None
    """Total contact force [N] per sensing object, shape ``(n_sensing,)``, dtype :class:`vec3`.
    ``None`` when ``measure_total=False``."""

    force_matrix: wp.array2d[wp.vec3] | None
    """Per-counterpart contact forces [N], shape ``(n_sensing, max_counterparts)``, dtype :class:`vec3`.
    Entry ``[i, j]`` is the force on sensing object ``i`` from counterpart ``counterpart_indices[i][j]``, in world
    frame. ``None`` when no counterparts are specified."""

    total_force_friction: wp.array[wp.vec3] | None
    """Total friction (tangential) contact force [N] per sensing object, shape ``(n_sensing,)``,
    dtype :class:`vec3`. ``None`` when ``measure_total=False``."""

    force_matrix_friction: wp.array2d[wp.vec3] | None
    """Per-counterpart friction (tangential) contact forces [N], shape ``(n_sensing, max_counterparts)``,
    dtype :class:`vec3`. Entry ``[i, j]`` is the friction force on sensing object ``i`` from counterpart
    ``counterpart_indices[i][j]``, in world frame. ``None`` when no counterparts are specified."""

    position_matrix: wp.array2d[wp.vec3] | None
    """Average contact positions [m] per counterpart, shape ``(n_sensing, max_counterparts)``, dtype :class:`vec3`.
    Entry ``[i, j]`` is the average world-frame interaction position between ``sensing_indices[i]`` and
    ``counterpart_indices[i][j]``. It averages the midpoint of all contacts between these objects, weighted by linear
    contact-force magnitude. Entries are zero when the interaction force is zero or :meth:`update` receives no body
    transforms. ``None`` when no counterparts are specified."""

    sensing_transforms: wp.array[wp.transform]
    """World-frame transforms of sensing objects [m, unitless quaternion],
    shape ``(n_sensing,)``, dtype :class:`transform`."""

    @property
    def sensing_obj_idx(self) -> list[int]:
        """Deprecated alias for :attr:`sensing_indices`.

        .. deprecated:: 1.4
            Use :attr:`sensing_indices` instead.
        """
        warnings.warn(_SENSING_OBJ_IDX_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.sensing_indices

    @sensing_obj_idx.setter
    def sensing_obj_idx(self, value: list[int]) -> None:
        warnings.warn(_SENSING_OBJ_IDX_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.sensing_indices = value

    @property
    def sensing_obj_type(self) -> Literal["body", "shape"]:
        """Deprecated alias for :attr:`sensing_type`.

        .. deprecated:: 1.4
            Use :attr:`sensing_type` instead.
        """
        warnings.warn(_SENSING_OBJ_TYPE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.sensing_type

    @sensing_obj_type.setter
    def sensing_obj_type(self, value: Literal["body", "shape"]) -> None:
        warnings.warn(_SENSING_OBJ_TYPE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.sensing_type = value

    @property
    def sensing_obj_transforms(self) -> wp.array[wp.transform]:
        """Deprecated alias for :attr:`sensing_transforms`.

        .. deprecated:: 1.4
            Use :attr:`sensing_transforms` instead.
        """
        warnings.warn(_SENSING_OBJ_TRANSFORMS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.sensing_transforms

    @sensing_obj_transforms.setter
    def sensing_obj_transforms(self, value: wp.array[wp.transform]) -> None:
        warnings.warn(_SENSING_OBJ_TRANSFORMS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.sensing_transforms = value

    def __init__(
        self,
        model: Model,
        *,
        sensing_bodies: str | list[str] | list[int] | None = None,
        sensing_shapes: str | list[str] | list[int] | None = None,
        counterpart_bodies: str | list[str] | list[int] | None = None,
        counterpart_shapes: str | list[str] | list[int] | None = None,
        measure_total: bool = True,
        verbose: bool | None = None,
        request_contact_attributes: bool = True,
        **kwargs: Any,
    ):
        """Initialize the SensorContact.

        Exactly one of ``sensing_bodies`` or ``sensing_shapes`` must be specified to define the sensing objects. At most
        one of ``counterpart_bodies`` or ``counterpart_shapes`` may be specified. If neither is specified, only
        :attr:`total_force` and :attr:`total_force_friction` are available (no per-counterpart breakdown or contact
        positions).

        Args:
            model: The simulation model providing shape/body definitions and world layout.
            sensing_bodies: List of body indices, single pattern to match against body labels, or list of patterns where
                any one matches.
            sensing_shapes: List of shape indices, single pattern to match against shape labels, or list of patterns
                where any one matches.
            counterpart_bodies: List of body indices, single pattern to match
                against body labels, or list of patterns where any one matches.
            counterpart_shapes: List of shape indices, single pattern to match
                against shape labels, or list of patterns where any one matches.
            measure_total: If True (default), :attr:`total_force` and :attr:`total_force_friction` are allocated.
                If False, both are None.
            verbose: If True, print details. If False, suppress details. If None, print details when
                ``wp.config.log_level`` is configured for debug logging.
            request_contact_attributes: If True (default), transparently request the extended contact attribute
                ``force`` from the model.
        """
        deprecated_sensing_bodies = kwargs.pop("sensing_obj_bodies", _UNSET)
        if deprecated_sensing_bodies is not _UNSET:
            warnings.warn(_SENSING_OBJ_BODIES_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            if sensing_bodies is not None and deprecated_sensing_bodies is not None:
                raise TypeError("Specify only one of `sensing_bodies` and deprecated `sensing_obj_bodies`.")
            if deprecated_sensing_bodies is not None:
                sensing_bodies = deprecated_sensing_bodies

        deprecated_sensing_shapes = kwargs.pop("sensing_obj_shapes", _UNSET)
        if deprecated_sensing_shapes is not _UNSET:
            warnings.warn(_SENSING_OBJ_SHAPES_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            if sensing_shapes is not None and deprecated_sensing_shapes is not None:
                raise TypeError("Specify only one of `sensing_shapes` and deprecated `sensing_obj_shapes`.")
            if deprecated_sensing_shapes is not None:
                sensing_shapes = deprecated_sensing_shapes

        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"SensorContact.__init__() got an unexpected keyword argument '{unexpected}'")

        if (sensing_bodies is None) == (sensing_shapes is None):
            raise ValueError("Exactly one of `sensing_bodies` and `sensing_shapes` must be specified")

        if (counterpart_bodies is not None) and (counterpart_shapes is not None):
            raise ValueError("At most one of `counterpart_bodies` and `counterpart_shapes` may be specified.")

        self.device = model.device
        self.verbose = verbose if verbose is not None else wp.config.log_level <= wp.LOG_DEBUG

        # request contact force attribute
        if request_contact_attributes:
            model.request_contact_attributes("force")

        if sensing_bodies is not None:
            s_bodies = match_labels(model.body_label, sensing_bodies)
            _check_index_bounds(s_bodies, len(model.body_label), "sensing_bodies", "bodies")
            s_shapes = []
        else:
            s_bodies = []
            s_shapes = match_labels(model.shape_label, sensing_shapes)
            _check_index_bounds(s_shapes, len(model.shape_label), "sensing_shapes", "shapes")

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

        # Determine whether sensing and counterparts are body-level or shape-level.
        sensing_is_body = sensing_bodies is not None
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
            sensing_indices, "sensing_bodies" if sensing_is_body else "sensing_shapes"
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

        if measure_total:
            self.total_force = wp.zeros(n_rows, dtype=wp.vec3, device=self.device)
            self.total_force_friction = wp.zeros(n_rows, dtype=wp.vec3, device=self.device)
        else:
            self.total_force = None
            self.total_force_friction = None

        if max_readings > 0:
            self.force_matrix = wp.zeros((n_rows, max_readings), dtype=wp.vec3, device=self.device)
            self.force_matrix_friction = wp.zeros((n_rows, max_readings), dtype=wp.vec3, device=self.device)
            self.position_matrix = wp.zeros((n_rows, max_readings), dtype=wp.vec3, device=self.device)
            self._position_weight = wp.zeros((n_rows, max_readings), dtype=wp.float32, device=self.device)
        else:
            self.force_matrix = None
            self.force_matrix_friction = None
            self.position_matrix = None
            self._position_weight = None

        self.sensing_type = "body" if sensing_is_body else "shape"
        self.counterpart_type = "body" if counterpart_is_body else ("shape" if counterpart_indices else None)
        self.sensing_indices = sensing_indices_ordered

        # Map each sensing object to its world's counterpart list.
        world_starts = np.array(sensing_world_start[:world_count])
        worlds = np.searchsorted(world_starts, sensing_indices_ordered, side="right") - 1
        self.counterpart_indices = [counterparts_by_world[w] for w in worlds]

        if self.verbose:
            print("SensorContact initialized:")
            print(f"  Sensing objects: {n_rows} ({self.sensing_type}s)")
            print(
                f"  Counterpart columns: {max_readings}"
                + (f" ({self.counterpart_type}s)" if self.counterpart_type else "")
            )
            print(
                f"  total_force: {'yes' if measure_total else 'no'}, "
                f"force_matrix: {'yes' if max_readings > 0 else 'no'}"
            )

        self._model = model
        self._sensing_indices = wp.array(sensing_indices_ordered, dtype=wp.int32, device=self.device)
        sensing_kind = _SENSING_KIND_BODY if sensing_is_body else _SENSING_KIND_SHAPE
        self._sensing_kinds = wp.full(n_rows, sensing_kind, dtype=wp.int32, device=self.device)
        self.sensing_transforms = wp.zeros(n_rows, dtype=wp.transform, device=self.device)

    def update(self, state: State | None, contacts: Contacts):
        """Update the contact sensor readings based on the provided state and contacts.

        Computes world-frame transforms for all sensing objects and evaluates contact forces and their friction
        (tangential) components (total and/or per-counterpart, depending on sensor configuration). When ``state``
        provides body transforms, also computes force-weighted per-counterpart contact positions.

        Args:
            state: The simulation state providing body transforms. If None (or a state without ``body_q``),
                :attr:`sensing_transforms` is left unchanged and :attr:`position_matrix` is reset to zero.
                Contact-force outputs are updated in either case.
            contacts: The contact data to evaluate.

        Raises:
            ValueError: If ``contacts.force`` is None.
            ValueError: If ``contacts.device`` does not match the sensor's device.
        """
        # update sensing transforms
        n = len(self._sensing_indices)
        if n > 0 and state is not None and state.body_q is not None:
            wp.launch(
                compute_sensing_transforms_kernel,
                dim=n,
                inputs=[
                    self._sensing_indices,
                    self._sensing_kinds,
                    self._model.shape_body,
                    self._model.shape_transform,
                    state.body_q,
                ],
                outputs=[self.sensing_transforms],
                device=self.device,
            )

        if contacts.force is None:
            raise ValueError(
                "SensorContact requires a ``Contacts`` object with ``force`` allocated. "
                "Create ``SensorContact`` before ``Contacts`` for automatically requesting it."
            )
        if contacts.device != self.device:
            raise ValueError(f"Contacts device ({contacts.device}) does not match sensor device ({self.device}).")
        self._eval_forces(state, contacts)

    def _eval_forces(self, state: State | None, contacts: Contacts):
        """Recompute force outputs and, when ``state.body_q`` is available, contact positions."""
        if self.total_force is not None:
            self.total_force.zero_()
            self.total_force_friction.zero_()
        if self.force_matrix is not None:
            self.force_matrix.zero_()
            self.force_matrix_friction.zero_()
            # reset positions together with forces so entries never pair a fresh force with a stale position
            self.position_matrix.zero_()
            self._position_weight.zero_()
        update_contact_positions = self.position_matrix is not None and state is not None and state.body_q is not None
        wp.launch(
            accumulate_contact_forces_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.force,
                contacts.rigid_contact_normal,
                self._model.shape_body,
                # body_q and the two position outputs below must be all-None or all-set:
                # the kernel dereferences body_q only under `if position_matrix`
                state.body_q if update_contact_positions else None,
                self._sensing_shape_to_row,
                self._counterpart_shape_to_col,
            ],
            outputs=[
                self.force_matrix,
                self.total_force,
                self.force_matrix_friction,
                self.total_force_friction,
                self.position_matrix if update_contact_positions else None,
                self._position_weight if update_contact_positions else None,
            ],
            device=self.device,
        )
        if update_contact_positions:
            wp.launch(
                normalize_contact_positions_kernel,
                dim=self.position_matrix.shape,
                inputs=[self.position_matrix, self._position_weight],
                device=self.device,
            )
