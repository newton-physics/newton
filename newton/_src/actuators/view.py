# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import warp as wp
from warp.types import is_array

from ..sim.model import Model
from .actuator import Actuator


@wp.kernel
def _init_actuator_dof_pairs_kernel(
    actuator_dofs: wp.array[wp.uint32], sorted_dofs: wp.array[wp.uint32], sorted_entries: wp.array[int]
):
    entry = wp.tid()
    sorted_dofs[entry] = actuator_dofs[entry]
    sorted_entries[entry] = entry


@wp.kernel
def _actuator_dof_mapping_kernel(
    sorted_dofs: wp.array[wp.uint32],
    sorted_entries: wp.array[int],
    entry_count: int,
    selected_dofs: wp.array[int],
    entry_mapping: wp.array[int],
):
    selection_index = wp.tid()
    dof = selected_dofs[selection_index]
    entry = int(-1)
    if dof >= 0:
        query_dof = wp.uint32(dof)
        candidate = entry_count - 1
        # Warp clamps past-end searches; handle last-key matches directly.
        if query_dof < sorted_dofs[candidate]:
            candidate = wp.lower_bound(sorted_dofs, 0, entry_count, query_dof + wp.uint32(1)) - 1
        if candidate >= 0 and sorted_dofs[candidate] == query_dof:
            entry = sorted_entries[candidate]
    entry_mapping[selection_index] = entry


def _build_actuator_dof_mapping(actuator_dofs: wp.array[wp.uint32], selected_dofs: wp.array) -> wp.array:
    """Map selected global DOFs to actuator entries, using -1 for unowned DOFs."""
    device = selected_dofs.device
    entry_mapping = wp.empty(selected_dofs.shape, dtype=int, device=device)
    if not entry_mapping.size:
        return entry_mapping
    entry_count = actuator_dofs.size
    if not entry_count:
        entry_mapping.fill_(-1)
        return entry_mapping

    # Warp's pair sort requires twice the input length for its scratch buffers.
    sorted_dofs = wp.empty(2 * entry_count, dtype=wp.uint32, device=device)
    sorted_entries = wp.empty(2 * entry_count, dtype=int, device=device)
    wp.launch(
        _init_actuator_dof_pairs_kernel,
        dim=entry_count,
        inputs=[actuator_dofs],
        outputs=[sorted_dofs, sorted_entries],
        device=device,
    )
    # Stable sorting keeps equal DOFs in entry order, so the last match wins.
    wp.utils.radix_sort_pairs(sorted_dofs, sorted_entries, entry_count)
    wp.launch(
        _actuator_dof_mapping_kernel,
        dim=entry_mapping.size,
        inputs=[sorted_dofs, sorted_entries, entry_count, selected_dofs.flatten()],
        outputs=[entry_mapping.flatten()],
        device=device,
    )
    return entry_mapping


@wp.kernel
def _gather_kernel(src: wp.array[Any], mapping: wp.array[int], dst: wp.array[Any]):
    selection_index = wp.tid()
    index = mapping[selection_index]
    if index >= 0:
        dst[selection_index] = src[index]


def _make_scatter_kernel(values_ndim: int, mask_ndim: int):
    @wp.kernel
    def scatter(values: Any, mapping: wp.array[int], mask: Any, mask_stride: int, dst: wp.array[Any]):
        selection_index = wp.tid()
        index = mapping[selection_index]
        mask_index = selection_index // mask_stride
        enabled = False
        if wp.static(mask_ndim == 1):
            enabled = mask[mask_index]
        else:
            enabled = mask[mask_index // mask.shape[1], mask_index % mask.shape[1]]
        if enabled and index >= 0:
            if wp.static(values_ndim == 1):
                dst[index] = values[selection_index]
            elif wp.static(values_ndim == 2):
                dst[index] = values[selection_index // values.shape[1], selection_index % values.shape[1]]
            else:
                prefix_index = selection_index // values.shape[2]
                dst[index] = values[
                    prefix_index // values.shape[1], prefix_index % values.shape[1], selection_index % values.shape[2]
                ]

    return scatter


_SCATTER_KERNELS = {
    (values_ndim, mask_ndim): _make_scatter_kernel(values_ndim, mask_ndim)
    for values_ndim, mask_ndim in ((1, 1), (2, 1), (3, 1), (3, 2))
}


def _actuator_components(actuator: Actuator) -> tuple[Any, ...]:
    delay = (actuator.delay,) if actuator.delay is not None else ()
    return (actuator.drive, *delay, *actuator.clamping)


def _component_owner(component: Any, actuators: Iterable[Actuator], scope: str) -> Actuator:
    """Return the first actuator that owns this component."""
    for actuator in actuators:
        if any(component is entry for entry in _actuator_components(actuator)):
            return actuator
    raise ValueError(f"Expected a component owned by an actuator {scope}")


def _get_actuator_parameter(actuator: Actuator, component: Any, name: str) -> wp.array:
    if not any(component is entry for entry in _actuator_components(actuator)):
        raise ValueError("Expected a component owned by the actuator")
    if not isinstance(name, str) or name.startswith("_") or name in getattr(component, "SHARED_PARAMS", ()):
        raise ValueError(f"Parameter '{name}' must be a public per-entry array, excluding shared parameters")
    parameter = getattr(component, name, None)
    if not isinstance(parameter, wp.array) or parameter.shape != (actuator.num_actuators,):
        raise ValueError(f"Expected one-dimensional per-entry parameter '{name}' of length {actuator.num_actuators}")
    return parameter


def _validate_array(
    array: wp.array, *, device: wp.Device, shape: tuple[int, ...] | None = None, dtype=None, label="array"
):
    if not is_array(array):
        raise ValueError(f"Expected a Warp {label}")
    if array.device != device:
        raise ValueError(f"Expected {label} on device {device}, got {array.device}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected {label} shape {shape}, got {array.shape}")
    if dtype is not None and array.dtype != dtype:
        raise ValueError(f"Expected {label} dtype {dtype}, got {array.dtype}")


def _gather_values(source: wp.array, mapping: wp.array) -> wp.array:
    device = mapping.device
    _validate_array(source, device=device)
    if source.ndim != 1:
        raise ValueError("Expected a one-dimensional source array")
    out = wp.zeros(mapping.shape, dtype=source.dtype, device=device, requires_grad=source.requires_grad)
    if mapping.size:
        wp.launch(
            _gather_kernel,
            dim=mapping.size,
            inputs=[source, mapping.flatten()],
            outputs=[out.flatten()],
            device=device,
        )
    return out


def _scatter_values(values: wp.array, target: wp.array, mapping: wp.array, mask) -> None:
    device = mapping.device
    _validate_array(target, device=device)
    if target.ndim != 1:
        raise ValueError("Expected a one-dimensional target array")
    if not is_array(values):
        values = wp.array(values, dtype=target.dtype, device=device)
    _validate_array(values, device=device, shape=mapping.shape, dtype=target.dtype, label="values")
    if not is_array(mask):
        mask = wp.array(mask, dtype=bool, device=device)
    _validate_array(mask, device=device, dtype=wp.bool, label="Boolean mask")
    mask_shapes = {(mapping.shape[0],)}
    if mapping.ndim == 3:
        mask_shapes.add(mapping.shape[:2])
    if mask.shape not in mask_shapes:
        raise ValueError(f"Expected Boolean mask shape in {sorted(mask_shapes)}, got {mask.shape}")
    if not mapping.size:
        return
    wp.launch(
        _SCATTER_KERNELS[values.ndim, mask.ndim],
        dim=mapping.size,
        inputs=[
            values,
            mapping.flatten(),
            mask,
            mapping.size // mask.size,
        ],
        outputs=[target],
        device=device,
    )


class ActuatorView:
    """Access actuator parameters and joint arrays through a DOF selection.

    .. experimental::

    Parameters and attributes store one Warp element per entry in a one-dimensional
    array. Results have the selection's shape, preserving dtype and gradients.
    Missing entries, including selected DOFs outside the union of the supplied
    actuators' ``indices``, read as zero and are skipped on writes.
    Shared components use their first owner's DOF mapping; owners must map entries to the same DOFs.
    Attribute access uses a common mapping per frequency. Differing actuator role
    mappings make that frequency unavailable for attribute access.
    Keep index metadata immutable and separate from writable data.
    Setter values and masks must not alias the written array.
    Use getters to read values and setters to update them. See
    :ref:`attribute frequencies <actuator-view-frequencies>` for metadata and index mappings.

    Args:
        actuators: Actuators included in the selection.
        dof_indices: Unique global velocity-DOF indices, with one to three axes.
            Accepts ``int32`` and ``uint32`` arrays; non-negative values must fit ``int32``.
            Defaults to the sole actuator's indices. Use ``-1`` in an ``int32`` array for explicit padding.
            Indices must remain unchanged.

    Construct the view outside graph capture. Nonempty reads allocate arrays,
    requiring the CUDA memory pool during capture. Prepare joint-frequency
    access once per topology outside capture.
    """

    def __init__(
        self, actuators: list[Actuator], *, dof_indices: wp.array[int] | wp.array[wp.uint32] | None = None
    ) -> None:
        if not actuators:
            raise ValueError("Expected at least one actuator")
        if dof_indices is None:
            if len(actuators) != 1:
                raise ValueError("Specify dof_indices for multiple actuators")
            dof_indices = actuators[0].indices
        elif (
            not is_array(dof_indices)
            or not 1 <= dof_indices.ndim <= 3
            or dof_indices.dtype not in (wp.int32, wp.uint32)
        ):
            raise ValueError("Expected an int32 or uint32 dof_indices array with one to three axes")
        if dof_indices.device.is_capturing:
            raise RuntimeError("Construct actuator views outside CUDA graph capture")
        selected = dof_indices.numpy().reshape(-1)
        valid = selected[selected >= 0]
        if np.any(selected < -1) or np.any(valid > np.iinfo(np.int32).max) or len(np.unique(valid)) != len(valid):
            raise ValueError(
                "Expected globally unique non-negative DOF indices within int32 range, with -1 for absent entries"
            )
        self.shape = dof_indices.shape
        """The caller's selection shape, without parameter element dimensions."""
        self.device = dof_indices.device
        self._attribute_bounds = {}
        selection = wp.array(selected.reshape(self.shape), dtype=int, device=self.device)
        mappings = {}
        for actuator in actuators:
            if actuator.indices.device != dof_indices.device:
                raise ValueError(
                    f"Expected actuator indices on device {dof_indices.device}, got {actuator.indices.device}"
                )
            mappings[actuator] = _build_actuator_dof_mapping(actuator.indices, selection)
        self._mappings = mappings
        # Attribute access treats selected DOFs that no included actuator drives like -1 padding.
        driven = np.any([entries.numpy() >= 0 for entries in mappings.values()], axis=0)
        self._dof_indices = self._make_mapping(np.where(driven, selected.reshape(self.shape).astype(np.int64), -1))
        self.actuators = tuple(mappings)
        """Actuators included in the view, in constructor order."""
        self.dof_indices = dof_indices
        """Selected global velocity-DOF indices; must remain unchanged."""
        self._full_mask = wp.ones(self.shape[0], dtype=bool, device=self.device)
        self._frequency_mappings = {}
        # Compare role mappings outside capture, which forbids reading index arrays back to the host.
        for frequency, attributes in (
            (Model.AttributeFrequency.JOINT_DOF, ("effort_indices",)),
            (Model.AttributeFrequency.JOINT_COORD, ("pos_indices", "target_pos_indices")),
        ):
            bindings = tuple(
                (entries, getattr(actuator, attribute))
                for actuator, entries in mappings.items()
                for attribute in attributes
            )
            initial = selected.reshape(self.shape) if frequency == Model.AttributeFrequency.JOINT_DOF else None
            try:
                indices = self._create_frequency_indices(bindings, initial=initial)
                mapping = self._dof_indices if initial is not None else self._make_mapping(indices)
            except ValueError as error:
                mapping = str(error)
            self._frequency_mappings[frequency] = mapping
        self._joint_mappings = {}

    @staticmethod
    def _metadata(source, name):
        return source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)

    def _attribute_frequency(self, name: str, source: Any):
        lookup = self._metadata(source, "get_attribute_frequency")
        if lookup is not None:
            try:
                return lookup(name)
            except KeyError:
                pass
        spec = Model._CORE_ATTRIBUTE_SPECS.get(name)
        return None if spec is None else spec.frequency

    def _attribute_access(self, name: str, source: Any) -> tuple[wp.array, wp.array]:
        frequency = self._attribute_frequency(name.replace(".", ":"), source)
        if frequency == Model.AttributeFrequency.JOINT:
            mapping = self._joint_mapping(source)
        else:
            mapping = self._frequency_mappings.get(frequency)
            if mapping is None:
                raise AttributeError(
                    f"Attribute '{name}' is not part of a supported joint selection (frequency {frequency!r})"
                )
        if isinstance(mapping, str):
            raise AttributeError(f"Attribute '{name}' (frequency {frequency!r}): {mapping}")
        return self._array(source, name), mapping

    def _joint_mapping(self, source: Any) -> wp.array:
        starts = self._metadata(source, "joint_qd_start")
        if not is_array(starts) or starts.ndim != 1 or starts.dtype is not wp.int32:
            raise AttributeError("Joint attributes require source.joint_qd_start as a one-dimensional int32 array")
        _validate_array(starts, device=self.device)
        mapping = self._joint_mappings.get(starts)
        if mapping is None:
            if self.device.is_capturing:
                raise RuntimeError("Prepare joint-frequency access once per topology outside CUDA graph capture")
            offsets = starts.numpy()
            if not offsets.size or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
                raise ValueError("Expected joint_qd_start to begin at zero and contain nondecreasing DOF offsets")
            selected = self._dof_indices.numpy()
            present = selected >= 0
            if np.any(selected[present] >= offsets[-1]):
                raise ValueError(f"Expected DOF indices below {offsets[-1]} from source.joint_qd_start")
            joint_indices = np.full(self.shape, -1, dtype=np.int32)
            joint_indices[present] = np.searchsorted(offsets, selected[present], side="right") - 1
            mapping = self._make_mapping(joint_indices)
            self._joint_mappings[starts] = mapping
        return mapping

    def get_actuator_dof_mapping(self, actuator: Actuator) -> wp.array[int]:
        """Return entry indices shaped like the selection; ``-1`` means unowned.

        The returned mapping must remain unchanged. Repeated actuator DOFs use the last entry, matching ArticulationView.

        Args:
            actuator: Actuator in this view.
        """
        return self._mappings[actuator]

    def get_actuator_parameter(self, name: str, component: Any) -> wp.array:
        """Read a component parameter in the selection's layout.

        Args:
            name: Parameter name on ``component``.
            component: Component object owned by an actuator in this view,
                such as ``actuator.drive`` or ``actuator.clamping[0]``.

        Raises:
            ValueError: The component has no owner in this view,
                or the attribute is not a public per-entry parameter.
        """
        actuator = _component_owner(component, self._mappings, "in this view")
        parameter = _get_actuator_parameter(actuator, component, name)
        return self._gather(parameter, self._mappings[actuator])

    def set_actuator_parameter(self, name: str, component: Any, values: wp.array, *, mask=None) -> None:
        """Write a per-entry parameter; repeated actuator DOFs write only the last entry.

        Args:
            name: Parameter name on ``component``.
            component: Owning component object, as for the getter.
            values: Values shaped like the getter's result.
            mask: Boolean mask over the first selection axis, or the first two axes of a three-axis selection.

        Raises:
            ValueError: The component has no owner in this view,
                or the attribute or write does not satisfy the parameter contract.
        """
        actuator = _component_owner(component, self._mappings, "in this view")
        parameter = _get_actuator_parameter(actuator, component, name)
        self._scatter(values, parameter, self._mappings[actuator], mask)

    def get_attribute(self, name: str, source: Any) -> wp.array:
        """Read selected values from a simulation array.

        Args:
            name: Dotted or colon-separated attribute path.
            source: Object or mapping holding the attribute.

        Raises:
            AttributeError: The frequency is unsupported or its index mapping is unavailable.
        """
        array, mapping = self._attribute_access(name, source)
        return self._gather(array, mapping)

    def set_attribute(self, name: str, target: Any, values: wp.array, *, mask=None) -> None:
        """Write selected values to a simulation array.

        Args:
            name: Dotted or colon-separated attribute path.
            target: Object or mapping holding the attribute.
            values: Values shaped like the getter's result.
            mask: Boolean mask over the first selection axis, or the first two axes of a three-axis selection.

        Raises:
            AttributeError: The frequency is unsupported or its index mapping is unavailable.
            ValueError: Multiple selection slots write the same target element.
        """
        array, mapping = self._attribute_access(name, target)
        self._scatter(values, array, mapping, mask)

    @staticmethod
    def _array(source: Any, name: str) -> wp.array:
        value = source
        if isinstance(value, Mapping) and name in value:
            value = value[name]
        else:
            for part in name.replace(":", ".").split("."):
                value = value[part] if isinstance(value, Mapping) else getattr(value, part)
        if not is_array(value):
            raise ValueError(f"Expected a Warp array for '{name}'")
        return value

    def _create_frequency_indices(
        self, bindings: tuple[tuple[wp.array, wp.array], ...], *, initial: np.ndarray | None = None
    ) -> np.ndarray:
        attribute_indices = np.full(self.shape, -1, dtype=np.int32) if initial is None else initial.copy()
        for entries, indices in bindings:
            entry_indices = entries.numpy()
            owned = entry_indices >= 0
            source_indices = indices.numpy()[entry_indices[owned]].astype(np.int64, copy=False)
            if np.any(source_indices < 0) or np.any(source_indices > np.iinfo(np.int32).max):
                raise ValueError("Attribute indices must be non-negative signed-32-bit integers")
            previous = attribute_indices[owned]
            if np.any((previous >= 0) & (previous != source_indices)):
                raise ValueError("Actuator index mappings disagree for the selected DOFs")
            attribute_indices[owned] = source_indices
        return attribute_indices

    def _make_mapping(self, indices: np.ndarray) -> wp.array:
        mapping = wp.array(indices, dtype=int, device=self.device)
        present = indices[indices >= 0]
        self._attribute_bounds[mapping] = (
            int(present.max()) + 1 if present.size else 0,
            len(np.unique(present)) != len(present),
        )
        return mapping

    def _validate_external_bounds(self, array: wp.array, mapping: wp.array, *, writing: bool) -> None:
        bounds = self._attribute_bounds.get(mapping)
        if bounds is None:
            return
        required, duplicates = bounds
        if array.shape[0] < required:
            raise ValueError(f"Expected at least {required} elements for this actuator mapping, got {array.shape[0]}")
        if writing and duplicates:
            raise ValueError("Cannot replace values through a mapping with duplicate output indices")

    def _gather(self, source: wp.array, mapping: wp.array) -> wp.array:
        self._validate_external_bounds(source, mapping, writing=False)
        return _gather_values(source, mapping)

    def _scatter(self, values: wp.array, target: wp.array, mapping: wp.array, mask) -> None:
        self._validate_external_bounds(target, mapping, writing=True)
        _scatter_values(
            values,
            target,
            mapping,
            self._full_mask if mask is None else mask,
        )
