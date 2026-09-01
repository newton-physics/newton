# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp
from warp.types import is_array

from .actuator import Actuator


@wp.kernel
def _build_mapping_kernel(
    actuator_indices: wp.array[wp.uint32],
    dof_indices: wp.array2d[int],
    actuator_count_per_world: int,
    mapping: wp.array2d[int],
):
    """Map selected global velocity DOFs to actuator parameter indices."""
    world, dof = wp.tid()
    start = world * actuator_count_per_world
    for local_index in range(actuator_count_per_world):
        actuator_index = start + local_index
        if int(actuator_indices[actuator_index]) == dof_indices[world, dof]:
            mapping[world, dof] = actuator_index


@wp.kernel
def _gather_parameter_kernel(src: Any, mapping: wp.array2d[int], dst: Any):
    """Gather actuator parameters into a view-shaped array."""
    world, dof = wp.tid()
    index = mapping[world, dof]
    if index >= 0:
        dst[world, dof] = src[index]


@wp.kernel
def _scatter_parameter_kernel(values: Any, mapping: wp.array2d[int], mask: wp.array[bool], dst: Any):
    """Scatter view-shaped actuator parameters for selected worlds."""
    world, dof = wp.tid()
    if mask[world]:
        index = mapping[world, dof]
        if index >= 0:
            dst[index] = values[world, dof]


class ActuatorView:
    """Read and write actuator parameters in a selected velocity-DOF layout.

    This view is independent of :class:`~newton.Model`. It derives mappings
    between its selected global velocity DOFs and each actuator's parameters.

    Args:
        actuators: Actuators to expose through the view.
        dof_indices: Global velocity-DOF indices for each selected column,
            shaped ``(world_count, dofs_per_world)``. Each actuator's
            parameter arrays must contain the same number of entries per world.
    """

    def __init__(self, actuators: list[Actuator], dof_indices: wp.array2d[int]) -> None:
        if not actuators:
            raise ValueError("Expected at least one actuator")
        if not is_array(dof_indices) or dof_indices.ndim != 2 or dof_indices.dtype is not wp.int32:
            raise ValueError("Expected a two-dimensional integer dof_indices array")
        if dof_indices.shape[0] == 0:
            raise ValueError("Expected dof_indices for at least one world")

        mappings = {}
        for actuator in actuators:
            if actuator.indices.device != dof_indices.device:
                raise ValueError(
                    f"Expected actuator indices on device {dof_indices.device}, got {actuator.indices.device}"
                )
            if actuator.indices.shape[0] % dof_indices.shape[0] != 0:
                raise ValueError("Expected each actuator to have the same number of entries per world")
            mapping = wp.full(dof_indices.shape, -1, dtype=int, device=dof_indices.device)
            if dof_indices.shape[1] != 0:
                wp.launch(
                    _build_mapping_kernel,
                    dim=dof_indices.shape,
                    inputs=[actuator.indices, dof_indices, actuator.indices.shape[0] // dof_indices.shape[0]],
                    outputs=[mapping],
                    device=dof_indices.device,
                )
            mappings[actuator] = mapping
        self._set_mappings(mappings)

    @classmethod
    def _from_mappings(cls, mappings: dict[Actuator, wp.array2d[int]]) -> ActuatorView:
        view = cls.__new__(cls)
        view._set_mappings(mappings)
        return view

    def _set_mappings(self, mappings: dict[Actuator, wp.array2d[int]]) -> None:
        if not mappings:
            raise ValueError("Expected at least one actuator")
        self._mappings = dict(mappings)
        mapping = next(iter(self._mappings.values()))
        self._full_mask = wp.full(mapping.shape[0], True, dtype=bool, device=mapping.device)

    def get_actuator_dof_mapping(self, actuator: Actuator) -> wp.array2d[int]:
        """Get the view-local DOF mapping for an actuator.

        Args:
            actuator: Actuator whose mapping to return.

        Returns:
            Mapping from selected velocity-DOF columns to actuator parameter
            indices. Unmapped columns contain ``-1``.
        """
        return self._mappings[actuator]

    def get_actuator_parameter(self, actuator: Actuator, component: Any, name: str) -> wp.array2d[Any]:
        """Read an actuator parameter in the selected velocity-DOF layout.

        Args:
            actuator: Actuator containing the parameter.
            component: Component object or actuator-relative string path,
                such as ``"controller"`` or ``"clamping.0"``.
            name: Parameter attribute name on the component.

        Returns:
            Parameter values in the selected parameter's units, shaped
            ``(world_count, dofs_per_world)``. Unmapped DOFs are zero.
        """
        mapping = self._mappings[actuator]
        if mapping.shape[1] == 0:
            return wp.empty(mapping.shape, dtype=float, device=mapping.device)
        parameter = self._get_parameter(actuator, component, name)
        self._validate_parameter_device(parameter, mapping)
        values = wp.zeros(mapping.shape, dtype=parameter.dtype, device=mapping.device)
        wp.launch(
            _gather_parameter_kernel,
            dim=mapping.shape,
            inputs=[parameter, mapping],
            outputs=[values],
            device=mapping.device,
        )
        return values

    def set_actuator_parameter(
        self,
        actuator: Actuator,
        component: Any,
        name: str,
        values: wp.array,
        mask=None,
    ) -> None:
        """Write an actuator parameter from the selected velocity-DOF layout.

        Args:
            actuator: Actuator containing the parameter.
            component: Component object or actuator-relative string path,
                such as ``"controller"`` or ``"clamping.0"``.
            name: Parameter attribute name on the component.
            values: Parameter values in the selected parameter's units, shaped
                ``(world_count, dofs_per_world)``.
            mask: Per-world mask. All worlds are updated when omitted.
        """
        mapping = self._mappings[actuator]
        mask = self._resolve_mask(mask, mapping)
        if mapping.shape[1] == 0:
            return

        parameter = self._get_parameter(actuator, component, name)
        self._validate_parameter_device(parameter, mapping)
        expected_shape = (*mapping.shape, *parameter.shape[1:])
        if not is_array(values):
            values = wp.array(values, dtype=parameter.dtype, shape=expected_shape, device=mapping.device, copy=False)
        if values.shape != expected_shape:
            raise ValueError(f"Expected values shape {expected_shape}, got {values.shape}")
        if values.device != mapping.device:
            raise ValueError(f"Expected values on device {mapping.device}, got {values.device}")

        wp.launch(
            _scatter_parameter_kernel,
            dim=mapping.shape,
            inputs=[values, mapping, mask],
            outputs=[parameter],
            device=mapping.device,
        )

    @staticmethod
    def _get_parameter(actuator: Actuator, component: Any, name: str) -> wp.array[Any]:
        if isinstance(component, str):
            component_name, separator, index = component.partition(".")
            component = getattr(actuator, component_name)
            if separator:
                component = component[int(index)]
        return getattr(component, name)

    def _resolve_mask(self, mask: Any, mapping: wp.array2d[int]) -> wp.array[bool]:
        if mask is None:
            return self._full_mask
        if not isinstance(mask, wp.array):
            try:
                return wp.array(mask, dtype=bool, shape=(mapping.shape[0],), device=mapping.device, copy=False)
            except Exception as error:
                raise ValueError(f"Expected Boolean mask with shape ({mapping.shape[0]},)") from error
        if mask.dtype is not wp.bool:
            raise ValueError(f"Expected Boolean mask, got dtype {mask.dtype}")
        if mask.shape != (mapping.shape[0],):
            raise ValueError(f"Expected mask shape ({mapping.shape[0]},), got {mask.shape}")
        if mask.device != mapping.device:
            raise ValueError(f"Expected mask on device {mapping.device}, got {mask.device}")
        return mask

    @staticmethod
    def _validate_parameter_device(parameter: wp.array[Any], mapping: wp.array2d[int]) -> None:
        if parameter.device != mapping.device:
            raise ValueError(f"Expected parameter on device {mapping.device}, got {parameter.device}")
