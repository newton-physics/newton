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

from dataclasses import dataclass

import numpy as np
import warp as wp

from .types import Descriptor, override

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

DEFAULT_DENSITY = 1000.0
"""
The global default density for materials, in kg/m^3.\n
Equals ``1000.0`` kg/m^3.
"""

DEFAULT_RESTITUTION = 0.0
"""
The global default restitution coefficient for material pairs.
Equals ``0.0``.
"""

DEFAULT_FRICTION = 0.7
"""
The global default friction coefficient for material pairs.
Equals ``0.7``.
"""

###
# Containers
###


@dataclass
class MaterialDescriptor(Descriptor):
    """
    A container to represent a managed material.

    Attributes:
        name (`str`): The name of the material.
        uid (`str`): The unique identifier (UUID) of the material.
        density (`float`): The density of the material, in kg/m^3.\n
            Defaults to the global default of ``1000.0`` kg/m^3.
        restitution (`float`): The coefficient of restitution,
            according to the Newtonian impact model.\n
            Defaults to the global default of ``0.0``.
        static_friction (`float`): The coefficient of static friction,
            according to the Coulomb friction model.\n
            Defaults to the global default of ``0.7``.
        dynamic_friction (`float`): The coefficient of dynamic friction,
            according to the Coulomb friction model.\n
            Defaults to the global default of ``0.7``.
        wid (`int`): Index of the world to which the material belongs.\n
            Defaults to `-1`, indicating that the material has not yet been added to a world.
        mid (`int`): Index of the material w.r.t. the world.\n
            Defaults to `-1`, indicating that the material has not yet been added to a world.
    """

    ###
    # Attributes
    ###

    density: float = DEFAULT_DENSITY
    """
    The density of the material, in kg/m^3.\n
    Defaults to the global default of ``1000.0`` kg/m^3.
    """

    restitution: float = DEFAULT_RESTITUTION
    """
    The coefficient of restitution, according to the Newtonian impact model.\n
    Defaults to the global default of ``0.0``.
    """

    static_friction: float = DEFAULT_FRICTION
    """
    The coefficient of static friction, according to the isotropic Coulomb friction model.\n
    Defaults to the global default of ``0.7``.
    """

    dynamic_friction: float = DEFAULT_FRICTION
    """
    The coefficient of dynamic friction, according to the isotropicCoulomb friction model.\n
    Defaults to the global default of ``0.7``.
    """

    ###
    # Metadata - to be set by the WorldDescriptor when added
    ###

    wid: int = -1
    """
    Index of the world to which the material belongs.\n
    Defaults to `-1`, indicating that the material has not yet been added to a world.
    """

    mid: int = -1
    """
    Index of the material w.r.t. the world.\n
    Defaults to `-1`, indicating that the material has not yet been added to a world.
    """

    @override
    def __repr__(self) -> str:
        """Returns a human-readable string representation of the MaterialDescriptor."""
        return (
            f"MaterialDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"density: {self.density},\n"
            f"restitution: {self.restitution},\n"
            f"static_friction: {self.static_friction},\n"
            f"dynamic_friction: {self.dynamic_friction}\n"
            f"wid: {self.wid},\n"
            f"mid: {self.mid},\n"
            f")"
        )


@dataclass
class MaterialPairProperties:
    """
    A container to represent the properties of a pair of materials, including friction and restitution coefficients.

    Attributes:
        restitution (`float`): The coefficient of restitution,
            according to the Newtonian impact model.\n
            Defaults to the global default of ``0.0``.
        static_friction (`float`): The coefficient of static surface friction,
            according to the Coulomb friction model.\n
            Defaults to the global default of ``0.7``.
        dynamic_friction (`float`): The coefficient of dynamic surface friction,
            according to the Coulomb friction model.\n
            Defaults to the global default of ``0.7``.
    """

    restitution: float = DEFAULT_RESTITUTION
    """
    The coefficient of restitution, according to the Newtonian impact model.\n
    Defaults to the global default of ``0.0``.
    """

    static_friction: float = DEFAULT_FRICTION
    """
    The coefficient of static surface friction, according to the Coulomb friction model.\n
    Defaults to the global default of ``0.7``.
    """

    dynamic_friction: float = DEFAULT_FRICTION
    """
    The coefficient of dynamic surface friction, according to the Coulomb friction model.\n
    Defaults to the global default of ``0.7``.
    """


class MaterialManager:
    """
    A class to manage materials used in simulations, including their properties and pair-wise interactions.

    Attributes:
        num_materials (int): The number of materials managed by this MaterialManager.
        materials (list[MaterialDescriptor]): A list of materials managed by this MaterialManager.
        pairs (list[list[MaterialPairProperties]]): A 2D list representing the properties of material pairs.
        default (MaterialDescriptor): The default material managed by this MaterialManager.
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
            default_material (MaterialDescriptor, optional): The default material to register.\n
                If None, a default material with the name 'default' will be created.
            default_static_friction (float): The default friction coefficient for material pairs.\n
                Defaults to `DEFAULT_FRICTION`.
            default_dynamic_friction (float): The default restitution coefficient for material pairs.\n
                Defaults to `DEFAULT_RESTITUTION`.
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
    def num_material_pairs(self) -> int:
        """
        Returns the number of material pairs managed by this MaterialManager.
        """
        N = len(self._materials)
        return N * (N + 1) // 2

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

    @property
    def default_pair(self) -> MaterialPairProperties:
        """
        Returns the properties of the default material pair managed by this MaterialManager.
        """
        return self._pair_properties[0][0]

    def has_material(self, name: str) -> bool:
        """
        Checks if a material with the given name exists in the manager.

        Args:
            name (str): The name of the material to check.

        Returns:
            bool: True if the material exists, False otherwise.
        """
        for m in self._materials:
            if m.name == name:
                return True
        return False

    def register(self, material: MaterialDescriptor) -> int:
        """
        Registers a new material with the manager.

        Args:
            material (MaterialDescriptor): The material descriptor to register.

        Returns:
            int: The index of the newly registered material.

        Raises:
            ValueError: If a material with the same name or UID already exists.
        """
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
        """
        Registers a new material pair with the manager.

        Args:
            first (MaterialDescriptor): The first material in the pair.
            second (MaterialDescriptor): The second material in the pair.
            material_pair (MaterialPairProperties): The properties of the material pair.

        Raises:
            ValueError: If either material is not already registered.
        """
        # Register the first material if it is not already registered
        if first.name not in [m.name for m in self.materials]:
            self.register(first)

        # Register the second material if it is not already registered
        if second.name not in [m.name for m in self.materials]:
            self.register(second)

        # Configure the material pair properties
        self.configure_pair(first=first.name, second=second.name, material_pair=material_pair)

    def configure_pair(self, first: int | str, second: int | str, material_pair: MaterialPairProperties):
        """
        Configures the properties of an existing material pair.

        Args:
            first (int | str): The index or name of the first material in the pair.
            second (int | str): The index or name of the second material in the pair.
            material_pair (MaterialPairProperties): The properties to set for the material pair.

        Raises:
            ValueError: If either material is not found.
        """
        # Get indices of the materials
        mid1 = self.index(first)
        mid2 = self.index(second)

        # Set the material pair properties
        self._pair_properties[mid1][mid2] = self._pair_properties[mid2][mid1] = material_pair

    def merge(self, other: "MaterialManager"):
        """
        Merges another MaterialManager into this one, combining their materials and material-pair properties.

        Args:
            other (MaterialManager): The other MaterialManager to merge.

        Raises:
            ValueError: If there are conflicting material names or UIDs.
        """
        # Iterate over the materials in the other manager
        for mat in other.materials:
            if not self.has_material(mat.name):
                self.register(mat)

        # Iterate over the material pairs in the other manager
        for i, mat1 in enumerate(other.materials):
            for j, mat2 in enumerate(other.materials):
                # Get the material pair properties from the other manager
                pair_props = other.pairs[i][j]
                # Configure the material pair properties in this manager if they exist
                if pair_props is not None:
                    self.configure_pair(first=mat1.name, second=mat2.name, material_pair=pair_props)

    def __getitem__(self, key) -> MaterialDescriptor:
        """
        Retrieves a material descriptor by its index or name.

        Args:
            key (str | int): The name or index of the material.

        Returns:
            MaterialDescriptor: The material descriptor.

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the material is not found.
        """
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
        """
        Retrieves the index of a material by its name or index.

        Args:
            key (str | int): The name or index of the material.

        Returns:
            int: The index of the material.

        Raises:
            ValueError: If the material is not found.
            TypeError: If the key is not a string or integer.
        """
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

        Returns:
            np.ndarray: A 2D numpy array containing restitution coefficients.
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

        Returns:
            np.ndarray: A 2D numpy array containing static friction coefficients.
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

        Returns:
            np.ndarray: A 2D numpy array containing dynamic friction coefficients.
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


@dataclass
class MaterialPairsModel:
    """
    A container to hold material-pairs properties for a simulation.

    Each material-pair property is stored as a flat array containing the unique
    non-zero elements of the lower-triangular part of a symmetric matrix, where
    the entry at row i and column j corresponds to the material pair (i, j). The
    indices correspond to the material indices defined by the MaterialManager.

    Attributes:
        num_pairs (int): Total number of material pairs in the model.
        restitution (wp.array): Restitution coefficients matrix for each material pair.\n
            Shape of ``(num_materials, num_materials)`` and type :class:`float`.
        static_friction (wp.array): Friction coefficients matrix for each material pair.\n
            Shape of ``(num_materials, num_materials)`` and type :class:`float`.
        dynamic_friction (wp.array): Friction coefficients matrix for each material pair.\n
            Shape of ``(num_materials, num_materials)`` and type :class:`float`.
    """

    num_pairs: int = 0
    """Total number of material pairs represented in the model."""

    # TODO: Switch to vec3f for including tangential restitution?
    restitution: wp.array | None = None
    """
    Restitution coefficients matrix for each material pair.\n
    Shape of ``(num_materials, num_materials)`` and type :class:`float`.
    """

    # TODO: Switch to vec3f for anisotropic+torsional friction?
    static_friction: wp.array | None = None
    """
    Friction coefficients matrix for each material pair.\n
    Shape of ``(num_materials, num_materials)`` and type :class:`float`.
    """

    # TODO: Switch to vec3f for anisotropic+torsional friction?
    dynamic_friction: wp.array | None = None
    """
    Friction coefficients matrix for each material pair.\n
    Shape of ``(num_materials, num_materials)`` and type :class:`float`.
    """
