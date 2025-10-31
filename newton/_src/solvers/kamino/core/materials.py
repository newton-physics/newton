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

"""
KAMINO: Material Model Types & Containers
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .types import Descriptor, float32

###
# Module interface
###

__all__ = [
    "DEFAULT_DENSITY",
    "DEFAULT_FRICTION",
    "DEFAULT_RESTITUTION",
    "MaterialDescriptor",
    "MaterialManager",
    "MaterialPairProperties",
    "MaterialPairsModel",
]

###
# Constants
###

# Global defaults
DEFAULT_DENSITY = 1000.0
"""The global default friction coefficient for material pairs."""
DEFAULT_FRICTION = 0.7
"""The global default friction coefficient for material pairs."""
DEFAULT_RESTITUTION = 0.0
"""The global default restitution coefficient for material pairs."""


###
# Containers
###


class MaterialDescriptor(Descriptor):
    """
    A container to represent a managed material.
    """

    def __init__(
        self,
        name: str = "",
        uid: str | None = None,
        density: float = DEFAULT_DENSITY,
        restitution: float = DEFAULT_RESTITUTION,
        static_friction: float = DEFAULT_FRICTION,
        dynamic_friction: float = DEFAULT_FRICTION,
    ):
        super().__init__(name, uid)

        self.wid: int = 0
        """Index of the world to which the material belongs."""

        self.mid: int = -1
        """Material index w.r.t. the corresponding world."""

        self.density: float = density
        """The density of the material, in kg/m^3."""

        self.restitution: float = restitution
        """The coefficient of restitution, according to the Newtonian impact model."""

        self.static_friction: float = static_friction
        """The coefficient of static friction, according to the Coulomb friction model."""

        self.dynamic_friction: float = dynamic_friction
        """The coefficient of dynamic friction, according to the Coulomb friction model."""

    def __repr__(self) -> str:
        return (
            f"MaterialDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"wid: {self.wid},\n"
            f"mid: {self.mid},\n"
            f"density: {self.density},\n"
            f"restitution: {self.restitution},\n"
            f"static_friction: {self.static_friction},\n"
            f"dynamic_friction: {self.dynamic_friction}\n"
            f")"
        )


class MaterialPairProperties:
    """
    A container to represent the properties of a pair of materials, including friction and restitution coefficients.

    Attributes:
        restitution (`float`): The coefficient of restitution, according to the Newtonian impact model.
        static_friction (`float`): The coefficient of static surface friction, according to the Coulomb friction model.
        dynamic_friction (`float`): The coefficient of dynamic surface friction, according to the Coulomb friction model.
    """

    def __init__(
        self,
        restitution: float = DEFAULT_RESTITUTION,
        static_friction: float = DEFAULT_FRICTION,
        dynamic_friction: float = DEFAULT_FRICTION,
    ):
        self.restitution: float = restitution
        self.static_friction: float = static_friction
        self.dynamic_friction: float = dynamic_friction


class MaterialManager:
    """
    A class to manage materials used in simulations, including their properties and pair-wise interactions.
    """

    def __init__(
        self,
        default_material: MaterialDescriptor | None = None,
        default_restitution: float = DEFAULT_RESTITUTION,
        default_static_friction: float = DEFAULT_FRICTION,
        default_dynamic_friction: float = DEFAULT_FRICTION,
    ):
        """
        Initializes the MaterialManager with an optional default material and its properties.

        Args:
            default_material (MaterialDescriptor, optional): The default material to register. If None, a default material with the name 'default' will be created.
            default_static_friction (float): The default friction coefficient for material pairs. Defaults to `DEFAULT_FRICTION`.
            default_dynamic_friction (float): The default restitution coefficient for material pairs. Defaults to `DEFAULT_RESTITUTION`.
        """
        # Declare the materials and material-pairs lists
        self._materials: list[MaterialDescriptor] = []
        self._pair_properties: list[list[MaterialPairProperties]] = []

        # Construct the default material if not provided
        if default_material is None:
            default_material = MaterialDescriptor("default")

        # Initialize a list of managed materials with the default material
        self.register(default_material)

        # Configure the default material pair properties
        self.register_pair(
            first=default_material,
            second=default_material,
            material_pair=MaterialPairProperties(
                restitution=default_restitution,
                static_friction=default_static_friction,
                dynamic_friction=default_dynamic_friction,
            ),
        )

    @property
    def num_materials(self) -> int:
        """
        Returns the number of materials managed by this MaterialManager.
        """
        return len(self._materials)

    @property
    def materials(self) -> list[MaterialDescriptor]:
        """
        Returns the list of materials managed by this MaterialManager.
        """
        return self._materials

    @property
    def pairs(self) -> list[list[MaterialPairProperties]]:
        """
        Returns the list of material-pair properties managed by this MaterialManager.
        """
        return self._pair_properties

    @property
    def default(self) -> MaterialDescriptor:
        """
        Returns the default material managed by this MaterialManager.
        """
        return self._materials[0]

    @default.setter
    def default(self, material: MaterialDescriptor):
        """
        Sets the default material to the provided material descriptor.

        Args:
            material (`MaterialDescriptor`): The material to set as the default.

        Raises:
            TypeError: If the provided material is not an instance of MaterialDescriptor.
        """
        if not isinstance(material, MaterialDescriptor):
            raise TypeError("`material` must be an instance of MaterialDescriptor.")
        self._materials[0] = material

    def register(self, material: MaterialDescriptor) -> int:
        # Get current bid from the number of bodies
        material.mid = self.num_materials

        # Check if the material already exists
        if material.name in [m.name for m in self.materials]:
            raise ValueError(f"Material name '{material.name}' already exists.")
        if material.uid in [m.uid for m in self.materials]:
            raise ValueError(f"Material UID '{material.uid}' already exists.")

        # Add the new material to the list of materials
        self.materials.append(material)

        # Add placeholder entries in the material pair properties list
        # NOTE: These are initialized to None and are to be set when the material pair is registered
        self._pair_properties.append([None] * (material.mid + 1))
        for i in range(material.mid):
            self._pair_properties[i].append(None)

        # Return the index of the new material
        return material.mid

    def register_pair(
        self, first: MaterialDescriptor, second: MaterialDescriptor, material_pair: MaterialPairProperties
    ):
        # Register the first material if it is not already registered
        if first.name not in [m.name for m in self.materials]:
            self.register(first)

        # Register the second material if it is not already registered
        if second.name not in [m.name for m in self.materials]:
            self.register(second)

        # Configure the material pair properties
        self.configure_pair(first=first.name, second=second.name, material_pair=material_pair)

    def configure_pair(self, first: int | str, second: int | str, material_pair: MaterialPairProperties):
        # Get indices of the materials
        mid1 = self.index(first)
        mid2 = self.index(second)

        # Set the material pair properties
        self._pair_properties[mid1][mid2] = self._pair_properties[mid2][mid1] = material_pair

    def __getitem__(self, key) -> MaterialDescriptor:
        # Check if the key is an integer
        if isinstance(key, int):
            # Check if the key is within the range of materials
            if key < 0 or key >= len(self.materials):
                raise IndexError(f"Material index '{key}' out of range.")
            # Return the material descriptor
            return self.materials[key]

        # Check if the key is a string
        elif isinstance(key, str):
            # Check if the key is a valid material name
            for m in self.materials:
                if m.name == key:
                    return m
            # If not found, raise an error
            raise ValueError(f"Material with name '{key}' not found.")

    def index(self, key: str | int) -> int:
        # Check if the name exists in the materials list
        if isinstance(key, str):
            for i in range(self.num_materials):
                if key == self.materials[i].name:
                    return i
        elif isinstance(key, int):
            # If the name is an integer, return it directly if it is a valid index
            if 0 <= key < self.num_materials:
                return key
        else:
            raise TypeError("Name argument must be a string or integer.")

        # If not found, raise an error
        raise ValueError(f"Material with key '{key}' not found.")

    def restitution_matrix(self) -> np.ndarray:
        """
        Generates a matrix of restitution coefficients for all material pairs.
        """
        # Get the number of materials
        N = len(self.materials)

        # Initialize the restitution matrix
        restitution = np.full((N, N), DEFAULT_RESTITUTION, dtype=np.float32)

        # Fill the matrix with the restitution coefficients
        for i in range(N):
            for j in range(N):
                # Check if the material pair properties exist
                if self._pair_properties[i][j] is not None:
                    restitution[i, j] = self._pair_properties[i][j].restitution
                else:
                    # Raise an error if the material pair properties are not set
                    raise ValueError(
                        f"Material-pair properties not set for materials:"
                        f"({self.materials[i].name}, {self.materials[j].name})"
                    )

        # Return the restitution matrix as a numpy array
        return restitution

    def static_friction_matrix(self) -> np.ndarray:
        """
        Generates a matrix of friction coefficients for all material pairs.
        """
        # Get the number of materials
        N = len(self.materials)

        # Initialize the friction matrix
        friction = np.full((N, N), DEFAULT_FRICTION, dtype=np.float32)

        # Fill the matrix with the friction coefficients
        for i in range(N):
            for j in range(N):
                # Check if the material pair properties exist
                if self._pair_properties[i][j] is not None:
                    friction[i, j] = self._pair_properties[i][j].static_friction
                else:
                    # Raise an error if the material pair properties are not set
                    raise ValueError(
                        f"Material-pair properties not set for materials:"
                        f"({self.materials[i].name}, {self.materials[j].name})"
                    )

        # Return the friction matrix as a numpy array
        return friction

    def dynamic_friction_matrix(self) -> np.ndarray:
        """
        Generates a matrix of friction coefficients for all material pairs.
        """
        # Get the number of materials
        N = len(self.materials)

        # Initialize the friction matrix
        friction = np.full((N, N), DEFAULT_FRICTION, dtype=np.float32)

        # Fill the matrix with the friction coefficients
        for i in range(N):
            for j in range(N):
                # Check if the material pair properties exist
                if self._pair_properties[i][j] is not None:
                    friction[i, j] = self._pair_properties[i][j].dynamic_friction
                else:
                    # Raise an error if the material pair properties are not set
                    raise ValueError(
                        f"Material-pair properties not set for materials:"
                        f"({self.materials[i].name}, {self.materials[j].name})"
                    )

        # Return the friction matrix as a numpy array
        return friction


class MaterialPairsModel:
    def __init__(self):
        self.num_pairs: int = 0
        """Total number of material pairs in the model."""

        # TODO: Switch to vec3f for including tangential restitution?
        self.restitution: wp.array2d(dtype=float32) | None = None
        """
        Restitution coefficients matrix for each material pair.\n
        Shape of ``(num_materials, num_materials)`` and type :class:`float32`.
        """

        # TODO: Switch to vec3f for anisotropic+torsional friction?
        self.static_friction: wp.array2d(dtype=float32) | None = None
        """
        Friction coefficients matrix for each material pair.\n
        Shape of ``(num_materials, num_materials)`` and type :class:`float32`.
        """

        # TODO: Switch to vec3f for anisotropic+torsional friction?
        self.dynamic_friction: wp.array2d(dtype=float32) | None = None
        """
        Friction coefficients matrix for each material pair.\n
        Shape of ``(num_materials, num_materials)`` and type :class:`float32`.
        """
