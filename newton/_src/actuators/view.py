# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp
from warp.types import is_array

from .actuator import Actuator, _gather_parameter_kernel, _scatter_parameter_kernel

if TYPE_CHECKING:
    from ..utils.selection import ArticulationView


class ActuatorView:
    """Read and write actuator parameters through view-local DOF mappings.

    This view is independent of :class:`~newton.Model`; callers provide the
    mapping from each actuator to the view's ``(world, DOF)`` layout.

    Args:
        mappings: Mapping from each actuator to its actuator parameter indices.
            Entries with value ``-1`` represent DOFs not driven by that actuator.
            All arrays must have the same shape and device.
    """

    def __init__(self, mappings: dict[Actuator, wp.array2d[int]]) -> None:
        self._mappings = dict(mappings)
        mapping = next(iter(self._mappings.values()))
        self._full_mask = wp.full(mapping.shape[0], True, dtype=bool, device=mapping.device)

    def get_actuator_dof_mapping(self, actuator: Actuator) -> wp.array2d[int]:
        """Get the view-local DOF mapping for an actuator.

        Args:
            actuator: Actuator whose mapping to return.

        Returns:
            Mapping from view DOFs to actuator parameter indices. Unmapped DOFs
            contain ``-1``.
        """
        return self._mappings[actuator]

    @classmethod
    def from_articulation_view(cls, articulation_view: ArticulationView, actuators: list[Actuator]) -> ActuatorView:
        """Build a view from an existing articulation view.

        Args:
            articulation_view: Articulation view that defines the desired DOF layout.
            actuators: Actuators to expose through the new view.

        Returns:
            A standalone actuator view.
        """
        return articulation_view.get_actuator_view(actuators)

    def get_actuator_parameter(self, actuator: Actuator, component_name: str, parameter_name: str) -> wp.array2d[Any]:
        """Read an actuator parameter in the view's DOF layout.

        Args:
            actuator: Actuator containing the parameter.
            component_name: Component attribute name, optionally followed by a
                list index, such as ``"controller"`` or ``"clamping.0"``.
            parameter_name: Parameter attribute name on the component.

        Returns:
            Parameter values in the selected parameter's units, shaped
            ``(world_count, dofs_per_world)``. Unmapped DOFs are zero.
        """
        parameter = self._get_parameter(actuator, component_name, parameter_name)
        return self._get_parameter_array(actuator, parameter)

    def _get_parameter_array(self, actuator: Actuator, parameter: wp.array[Any]) -> wp.array2d[Any]:
        mapping = self._mappings[actuator]
        if mapping.shape[1] == 0:
            return wp.empty(mapping.shape, dtype=parameter.dtype, device=mapping.device)
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
        component_name: str,
        parameter_name: str,
        values: wp.array2d[Any],
        mask: wp.array[bool] | None = None,
    ) -> None:
        """Write an actuator parameter from the view's DOF layout.

        Args:
            actuator: Actuator containing the parameter.
            component_name: Component attribute name, optionally followed by a
                list index, such as ``"controller"`` or ``"clamping.0"``.
            parameter_name: Parameter attribute name on the component.
            values: Parameter values in the selected parameter's units, shaped
                ``(world_count, dofs_per_world)``.
            mask: Per-world mask. All worlds are updated when omitted.
        """
        parameter = self._get_parameter(actuator, component_name, parameter_name)
        self._set_parameter_array(actuator, parameter, values, mask)

    def _set_parameter_array(
        self,
        actuator: Actuator,
        parameter: wp.array[Any],
        values: wp.array2d[Any],
        mask: wp.array[bool] | None,
    ) -> None:
        mapping = self._mappings[actuator]
        if mapping.shape[1] == 0:
            return

        expected_shape = (*mapping.shape, *parameter.shape[1:])
        if not is_array(values):
            values = wp.array(values, dtype=parameter.dtype, shape=expected_shape, device=mapping.device, copy=False)
        if values.shape[:2] != expected_shape[:2]:
            raise ValueError(f"Expected values shape {expected_shape}, got {values.shape}")

        if mask is None:
            mask = self._full_mask
        elif not isinstance(mask, wp.array):
            mask = wp.array(mask, dtype=bool, shape=(mapping.shape[0],), device=mapping.device, copy=False)
        if mask.shape != (mapping.shape[0],):
            raise ValueError(f"Expected mask shape ({mapping.shape[0]},), got {mask.shape}")

        wp.launch(
            _scatter_parameter_kernel,
            dim=mapping.shape,
            inputs=[values, mapping, mask],
            outputs=[parameter],
            device=mapping.device,
        )

    @staticmethod
    def _get_parameter(actuator: Actuator, component_name: str, parameter_name: str) -> wp.array[Any]:
        component_name, separator, index = component_name.partition(".")
        component = getattr(actuator, component_name)
        if separator:
            component = component[int(index)]
        return getattr(component, parameter_name)
