# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp

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

    @classmethod
    def from_articulation_view(cls, articulation_view: ArticulationView, actuators: list[Actuator]) -> ActuatorView:
        """Build a view from an existing articulation view.

        Args:
            articulation_view: Articulation view that defines the desired DOF layout.
            actuators: Actuators to expose through the new view.

        Returns:
            A standalone actuator view.
        """
        mappings = {
            actuator: articulation_view._get_actuator_dof_mapping(actuator).reshape((articulation_view.world_count, -1))
            for actuator in actuators
        }
        return cls(mappings)

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
        mapping = self._mappings[actuator]
        parameter = self._get_parameter(actuator, component_name, parameter_name)
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
        mapping = self._mappings[actuator]
        parameter = self._get_parameter(actuator, component_name, parameter_name)
        wp.launch(
            _scatter_parameter_kernel,
            dim=mapping.shape,
            inputs=[values, mapping, self._full_mask if mask is None else mask],
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
