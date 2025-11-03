# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

import warp as wp
from warp.types import matrix

from ..sim import Model


class mat32(matrix(shape=(3, 2), dtype=wp.float32)):
    pass


class SolverData:
    """
    Registry for time-varying simulation data that is not part of the State.

    Provides a solver-agnostic interface for accessing quantities computed during
    forward dynamics such as accelerations, contact forces, and constraint forces.
    Fields are organized by frequency (``body_``, ``shape_``, ``joint_``, ``contact_``, etc.) and
    must be explicitly required before use.

    See :ref:`SolverDataFields` in the documentation for detailed usage and field reference.
    """

    solver_fields: dict[str, int]
    """Fields supported by the solver."""

    custom_fields: dict[str, str]
    """Solver-defined field names, mapping to frequency."""

    required_fields: dict[str, bool]
    """Fields that have been marked as required, mapping each field name to a bool
    indicating whether it is active."""

    frequency_sizes: dict[str, int | None]
    """Mapping from frequency prefix to the count/size for that frequency.
    Runtime-determined frequencies (e.g., 'contact') are added on first use."""

    body_acceleration: wp.array(dtype=wp.spatial_vector)
    """Linear and angular acceleration of the body (COM-referenced) in world frame."""

    body_parent_joint_force: wp.array(dtype=wp.spatial_vector)
    """Parent joint force and torque."""

    contact_force_scalar: wp.array(dtype=float)
    """Magnitude of contact force."""
    contact_force_vector_c: wp.array(dtype=wp.vec3f)
    """Contact force vector in contact frame."""
    contact_torque_vector_c: wp.array(dtype=wp.vec3f)
    """Contact torque vector in contact frame."""
    contact_frame_w: wp.array(dtype=mat32)
    """Unit vectors z and x defining the contact frame in world frame, where z and x define the
    normal and first tangent directions, respectively. The second tangent is cross(z, x)."""

    def __init__(self, model: Model, data_fields: dict[str, int], verbose: bool | None = None):
        self.verbose = verbose if verbose is not None else wp.config.verbose

        self.model = model
        self.custom_fields = {}
        self.required_fields = {}
        self.solver_fields = data_fields

        # Initialize frequency sizes with known model-based frequencies
        self.frequency_sizes = {
            "body": model.body_count,
            "shape": model.shape_count,
            "joint": model.joint_count,
            "joint_dof": model.joint_dof_count,
            "joint_coord": model.joint_coord_count,
            "articulation": model.articulation_count,
            "contact": None,
        }

        # Verify known & extract unknown frequencies from solver
        for field_name, field_size in data_fields.items():
            frequency = self.find_attribute_frequency(field_name)
            # For unknown frequencies (like 'contact'), set size from first occurrence
            if frequency not in self.frequency_sizes:
                self.frequency_sizes[frequency] = field_size
                if self.verbose:
                    print(f"Setting frequency size from solver data fields: {frequency} = {field_size}")
            elif field_size != self.frequency_sizes[frequency]:
                raise ValueError(
                    f"Solver field '{field_name}' size {field_size} does not match "
                    f"expected {frequency} size {self.frequency_sizes[frequency]}"
                )

    def _require_fields(self, fields: dict[str, bool]):
        """If not allocated, allocate and zero-initialize fields"""
        if missing_fields := set(fields).difference(self.solver_fields):
            raise TypeError(
                f"Solver {self.__class__.__name__} does not support required data fields: {list(missing_fields)}"
            )

        for field in fields:
            if hasattr(self, field):
                continue
            frequency = self.find_attribute_frequency(field)
            field_size = self.frequency_sizes[frequency]

            if self.verbose:
                print(f"Initializing SolverData field {field} with size {field_size}")

            hint = typing.get_type_hints(self).get(field)
            if hint is None:
                raise TypeError(f"SolverData field {field} is not defined.")

            if isinstance(hint, wp.array):
                setattr(self, field, wp.zeros(field_size, dtype=hint.dtype, device=self.device))
            else:
                raise NotImplementedError(f"Field {field} has unimplemented type {hint}")
        self.required_fields.update(fields)

    def set_field_active(self, *fields, active=True):
        """Activate or deactivate fields. Deactivated fields remain allocated but are not computed."""
        if missing := set(fields).difference(self.required_fields):
            raise RuntimeError(f"Fields {missing} must be required before they can be (de-)activated.")
        self.required_fields.update(dict.fromkeys(fields, active))

    def register_custom_field(self, field_name: str, field_frequency: str, field_type: type) -> None:
        """Register a custom solver-specific field."""
        pass
        # TODO: implement

    def find_attribute_frequency(self, name: str):
        """
        Parse the frequency prefix of an attribute name.

        Args:
            name: Attribute name (e.g., "body_acceleration")

        Returns:
            Frequency prefix (e.g., "body", no underscore)

        Raises:
            AttributeError: If the attribute frequency is not known.
        """

        frequencies = tuple(self.frequency_sizes)

        if not name.startswith(frequencies):
            raise AttributeError(f"Attribute frequency of '{name}' is not known")

        return next(freq for freq in sorted(frequencies, key=len, reverse=True) if name.startswith(freq + "_"))

    @property
    def device(self) -> wp.context.Device:
        """Device used by the solver."""
        return self.model.device
