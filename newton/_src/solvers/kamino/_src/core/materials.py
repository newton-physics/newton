# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Mechanisms for defining and managing materials and their properties.

This module provides a set of data types and operations that realize configurable
material properties that can be queried at simulation runtime. It includes:

- :class:`MaterialDescriptor`: A container to represent a managed material.

- :class:`MaterialPairProperties`: A container to represent the properties of a pair
  of materials, including friction and restitution coefficients.

- :class:`MaterialManager`: A class to manage materials used in simulations, including
  their properties and pairwise interactions.

- :class:`MaterialsModel`: A container to hold and manage per-material properties.

- :class:`MaterialPairsModel`: A container to hold and manage per-material-pair properties.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import warp as wp

from .....core.types import override
from .....geometry.materials import find_material_pair, material_pair_key
from .types import Descriptor

logger = logging.getLogger(__name__)

###
# Module interface
###

__all__ = [
    "DEFAULT_FRICTION",
    "DEFAULT_RESTITUTION",
    "MaterialDescriptor",
    "MaterialManager",
    "MaterialPairProperties",
    "MaterialPairsModel",
    "MaterialsModel",
    "make_get_mixed_material_pair_property",
]

###
# Constants
###

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
# Types
###


class MaterialMixMode(IntEnum):
    """
    An enumeration defining the heuristic modes for deriving
    pairwise material properties from individual materials.

    This is used when no specific material-pair properties
    are defined and the properties must be derived.
    """

    AVERAGE = 0
    """Pairwise property is the average of the two material properties."""

    MULTIPLY = 1
    """Pairwise property is the product of the two material properties."""

    MAX = 2
    """Pairwise property is the maximum of the two material properties."""

    MIN = 3
    """Pairwise property is the minimum of the two material properties."""

    GEOMETRIC_AVERAGE = 4
    """Pairwise property is the geometric average (square root of the product) of the two material properties."""

    @classmethod
    def from_string(cls, s: str) -> MaterialMixMode:
        """Converts a string to a MaterialMixMode enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid MaterialMixMode: {s}. Valid options are: {[e.name for e in cls]}") from e

    @override
    def __str__(self):
        """Returns a string representation of the MaterialMixMode."""
        return f"MaterialMixMode.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the MaterialMixMode."""
        return self.__str__()


@dataclass
class MaterialDescriptor(Descriptor):
    """
    A container to represent a managed material.

    To support model specifications such as USD (which currently do not support material-pair
    definitions), this descriptor holds extrinsic properties of a material (e.g., friction, restitution),
    which are actually dependent on the pairwise interactions of the material with others.

    Attributes:
        name: The name of the material.
        uid: The unique identifier (UUID) of the material.
        restitution: The coefficient of restitution, according to the Newtonian impact model.
            Defaults to the global default of ``0.0``.
        static_friction: The coefficient of static friction, according to the Coulomb friction model.
            Defaults to the global default of ``0.7``.
        dynamic_friction: The coefficient of dynamic friction, according to the Coulomb friction model.
            Defaults to the global default of ``0.7``.
        wid: Index of the world to which the material belongs.
            Defaults to `-1`, indicating that the material has not yet been added to a world.
        mid: Index of the material w.r.t. the world.
            Defaults to `-1`, indicating that the material has not yet been added to a world.
    """

    ###
    # Attributes
    ###

    restitution: float = DEFAULT_RESTITUTION
    """
    The coefficient of restitution, according to the Newtonian impact model.
    Defaults to the global default of ``0.0``.
    """

    static_friction: float = DEFAULT_FRICTION
    """
    The coefficient of static friction, according to the isotropic Coulomb friction model.
    Defaults to the global default of ``0.7``.
    """

    dynamic_friction: float = DEFAULT_FRICTION
    """
    The coefficient of dynamic friction, according to the isotropic Coulomb friction model.
    Defaults to the global default of ``0.7``.
    """

    ###
    # Metadata - to be set by the WorldDescriptor when added
    ###

    wid: int = -1
    """
    Index of the world to which the material belongs.
    Defaults to `-1`, indicating that the material has not yet been added to a world.
    """

    mid: int = -1
    """
    Index of the material w.r.t. the world.
    Defaults to `-1`, indicating that the material has not yet been added to a world.
    """

    @override
    def __repr__(self) -> str:
        """Returns a human-readable string representation of the MaterialDescriptor."""
        return (
            f"MaterialDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
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
        restitution: The coefficient of restitution, according to the Newtonian impact model.
            Defaults to the global default of ``0.0``.
        static_friction: The coefficient of static surface friction, according to the Coulomb friction model.
            Defaults to the global default of ``0.7``.
        dynamic_friction: The coefficient of dynamic surface friction, according to the Coulomb friction model.
            Defaults to the global default of ``0.7``.
    """

    restitution: float = DEFAULT_RESTITUTION
    """
    The coefficient of restitution, according to the Newtonian impact model.
    Defaults to the global default of ``0.0``.
    """

    static_friction: float = DEFAULT_FRICTION
    """
    The coefficient of static surface friction, according to the Coulomb friction model.
    Defaults to the global default of ``0.7``.
    """

    dynamic_friction: float = DEFAULT_FRICTION
    """
    The coefficient of dynamic surface friction, according to the Coulomb friction model.
    Defaults to the global default of ``0.7``.
    """


###
# Containers
###


@dataclass
class MaterialsModel:
    """
    A container to hold and manage per-material properties.

    Each material property is stored as an array ordered according
    to the material index (`mid`) defined by the MaterialManager.

    Attributes:
        num_materials: Total number of materials represented in the model.
        restitution: Array of restitution coefficients for each registered material.
            Shape of ``(num_materials,)``.
        static_friction: Array of static friction coefficients for each registered material.
            Shape of ``(num_materials,)``.
        dynamic_friction: Array of dynamic friction coefficients for each registered material.
            Shape of ``(num_materials,)``.
    """

    num_materials: int = 0
    """Total number of materials represented in the model."""

    restitution: wp.array[wp.float32] | None = None
    """
    Array of restitution coefficients for each registered material.
    Shape of ``(num_materials,)``.
    """

    # TODO: Switch to wp.vec3f for anisotropic+torsional friction?
    static_friction: wp.array[wp.float32] | None = None
    """
    Array of static friction coefficients for each registered material.
    Shape of ``(num_materials,)``.
    """

    # TODO: Switch to wp.vec3f for anisotropic+torsional friction?
    dynamic_friction: wp.array[wp.float32] | None = None
    """
    Array of dynamic friction coefficients for each registered material.
    Shape of ``(num_materials,)``.
    """


@dataclass
class MaterialPairsModel:
    """Explicit material pair properties stored in sorted key order."""

    num_material_pairs: int = 0
    """Number of explicitly configured material pairs."""
    keys: wp.array[wp.uint64] | None = None
    """Sorted keys for the configured pairs, shape [num_material_pairs]."""
    restitution: wp.array[wp.float32] | None = None
    """Pair restitution coefficients, shape [num_material_pairs]."""
    static_friction: wp.array[wp.float32] | None = None
    """Pair static friction coefficients, shape [num_material_pairs]."""
    dynamic_friction: wp.array[wp.float32] | None = None
    """Pair dynamic friction coefficients, shape [num_material_pairs]."""


###
# Functions
###


@wp.func
def material_average(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the average of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The average of the two material property values.
    """
    return 0.5 * (value1 + value2)


@wp.func
def material_multiply(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the product of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The product of the two material property values.
    """
    return value1 * value2


@wp.func
def material_max(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the maximum of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The maximum of the two material property values.
    """
    return wp.max(value1, value2)


@wp.func
def material_min(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the minimum of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The minimum of the two material property values.
    """
    return wp.min(value1, value2)


@wp.func
def material_geometric_average(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the geometric average of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The geometric average of the two material property values.
    """
    return wp.sqrt(value1 * value2)


def get_material_mixing_function(
    mixmode: MaterialMixMode = MaterialMixMode.AVERAGE,
) -> Callable[[wp.float32, wp.float32], wp.float32]:
    """
    Returns the appropriate material mix function based on the specified mixing mode.
    """
    # Select the appropriate mixing function based on the mixing mode
    match mixmode:
        case MaterialMixMode.AVERAGE:
            mix_func = material_average
        case MaterialMixMode.MULTIPLY:
            mix_func = material_multiply
        case MaterialMixMode.MAX:
            mix_func = material_max
        case MaterialMixMode.MIN:
            mix_func = material_min
        case MaterialMixMode.GEOMETRIC_AVERAGE:
            mix_func = material_geometric_average
        case _:
            raise ValueError(f"Unsupported material mixing mode: {mixmode}")

    # Return the selected mixing function
    return mix_func


def make_get_mixed_material_pair_property(mixmode: MaterialMixMode = MaterialMixMode.AVERAGE):
    """
    Generates a Warp function to retrieve material pair
    property based on the specified mixing mode.

    Args:
        mixmode: The mixing mode to use for material pair property.

    Returns:
        A Warp function that retrieves material pair property.
    """
    # Select the appropriate mixing function based on the mixing mode
    mix_func = get_material_mixing_function(mixmode)

    # Define the Warp function to retrieve material pair properties
    @wp.func
    def _get_mixed_material_pair_property(
        property_0: wp.float32,
        property_1: wp.float32,
    ) -> wp.float32:
        """
        Retrieves the mixed properties of a material pair given the properties of the two materials.

        Args:
            property_0: The property of the first material.
            property_1: The property of the second material.

        Returns:
            The mixed property of the two materials.
        """
        return mix_func(property_0, property_1)

    # Return the generated Warp function
    return _get_mixed_material_pair_property


def make_get_material_pair_properties(mixmode: MaterialMixMode = MaterialMixMode.AVERAGE):
    """
    Generates a Warp function to retrieve material pair
    properties based on the specified mixing mode.

    Args:
        mixmode: The mixing mode to use for material pair properties.

    Returns:
        A Warp function that retrieves material pair properties.
    """
    # Select the appropriate mixing function based on the mixing mode
    mix_func = get_material_mixing_function(mixmode)

    # Define the Warp function to retrieve material pair properties
    @wp.func
    def _get_material_pair_properties(
        mid1: wp.int32,
        mid2: wp.int32,
        material_restitution: wp.array[wp.float32],
        material_static_friction: wp.array[wp.float32],
        material_dynamic_friction: wp.array[wp.float32],
        material_pair_keys: wp.array[wp.uint64],
        material_pair_restitution: wp.array[wp.float32],
        material_pair_static_friction: wp.array[wp.float32],
        material_pair_dynamic_friction: wp.array[wp.float32],
    ) -> tuple[wp.float32, wp.float32, wp.float32]:
        """
        Retrieves the properties of a material pair given their material indices.

        If material-pair properties are not defined (i.e., negative values) for the given
        material indices `mid1, mid2`, the properties are computed from the individual
        materials using the configured mixing method.

        Args:
            mid1: The index of the first material.
            mid2: The index of the second material.
            material_restitution: The per-material restitution coefficients.
            material_static_friction: The per-material static friction coefficients.
            material_dynamic_friction: The per-material dynamic friction coefficients.
            material_pair_keys: Sorted keys for the configured material pairs.
            material_pair_restitution: The per-material-pair restitution coefficients.
            material_pair_static_friction: The per-material-pair static friction coefficients.
            material_pair_dynamic_friction: The per-material-pair dynamic friction coefficients.

        Returns:
            The restitution, static friction, and dynamic friction coefficients for the material pair.
        """
        row = find_material_pair(material_pair_keys, mid1, mid2)
        restitution = -1.0
        static_friction = -1.0
        dynamic_friction = -1.0
        if row >= 0:
            restitution = material_pair_restitution[row]
            static_friction = material_pair_static_friction[row]
            dynamic_friction = material_pair_dynamic_friction[row]

        # If any property is negative, compute the material pair properties using the set mixing method
        if restitution < 0.0:
            restitution = mix_func(material_restitution[mid1], material_restitution[mid2])
        if static_friction < 0.0:
            static_friction = mix_func(material_static_friction[mid1], material_static_friction[mid2])
        if dynamic_friction < 0.0:
            dynamic_friction = mix_func(material_dynamic_friction[mid1], material_dynamic_friction[mid2])

        # Return the material pair properties
        return restitution, static_friction, dynamic_friction

    # Return the generated Warp function
    return _get_material_pair_properties


###
# Interfaces
###


class MaterialManager:
    """
    A class to manage materials used in simulations, including their properties and pairwise interactions.

    Attributes:
        num_materials: The number of materials managed by this MaterialManager.
        materials: A list of materials managed by this MaterialManager.
        pairs: A 2D list representing the properties of material pairs.
        default: The default material managed by this MaterialManager.
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
            default_material: The default material to register.
                If None, a default material with the name 'default' will be created.
            default_restitution: The default restitution coefficient for material pairs.
                Defaults to ``DEFAULT_RESTITUTION``.
            default_static_friction: The default static friction coefficient for material pairs.
                Defaults to ``DEFAULT_FRICTION``.
            default_dynamic_friction: The default dynamic friction coefficient for material pairs.
                Defaults to ``DEFAULT_FRICTION``.
        """
        # Declare the materials and material-pairs lists
        self._materials: list[MaterialDescriptor] = []
        self._material_indices: dict[str, int] = {}
        self._material_uids: set[str] = set()
        self._pair_properties: dict[int, MaterialPairProperties] = {}

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
        Returns the number of possible unordered pairs, including self-pairs.
        """
        N = len(self._materials)
        return N * (N + 1) // 2

    @property
    def num_configured_pairs(self) -> int:
        """Return the number of explicitly configured pairs."""
        return len(self._pair_properties)

    @property
    def materials(self) -> list[MaterialDescriptor]:
        """
        Returns the list of materials managed by this MaterialManager.
        """
        return self._materials

    @property
    def pairs(self) -> list[list[MaterialPairProperties | None]]:
        """Return a symmetric snapshot of the configured pairs for inspection."""
        pairs = [[None] * self.num_materials for _ in self.materials]
        for key, properties in self._pair_properties.items():
            first, second = key >> 32, key & 0xFFFFFFFF
            pairs[first][second] = pairs[second][first] = properties
        return pairs

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
            material: The material to set as the default.

        Raises:
            TypeError: If the provided material is not an instance of MaterialDescriptor.
        """
        if not isinstance(material, MaterialDescriptor):
            raise TypeError("`material` must be an instance of MaterialDescriptor.")
        previous = self._materials[0]
        self._validate_registration(material, replacing=previous)
        del self._material_indices[previous.name]
        self._material_uids.remove(previous.uid)
        material.mid = 0
        self._materials[0] = material
        self._material_indices[material.name] = 0
        self._material_uids.add(material.uid)

    @property
    def default_pair(self) -> MaterialPairProperties:
        """
        Returns the properties of the default material pair managed by this MaterialManager.
        """
        return self._pair_properties[0]

    def has_material(self, name: str) -> bool:
        """
        Checks if a material with the given name exists in the manager.

        Args:
            name: The name of the material to check.

        Returns:
            True if the material exists, False otherwise.
        """
        return name in self._material_indices

    def _validate_registration(self, material: MaterialDescriptor, replacing: MaterialDescriptor | None = None) -> None:
        if material.name in self._material_indices and (replacing is None or material.name != replacing.name):
            raise ValueError(f"Material name '{material.name}' already exists.")
        if material.uid in self._material_uids and (replacing is None or material.uid != replacing.uid):
            raise ValueError(f"Material UID '{material.uid}' already exists.")

    def register(self, material: MaterialDescriptor) -> int:
        """
        Registers a new material with the manager.

        Args:
            material: The material descriptor to register.

        Returns:
            The index of the newly registered material.

        Raises:
            ValueError: If a material with the same name or UID already exists.
        """
        self._validate_registration(material)
        material.mid = self.num_materials
        self._materials.append(material)
        self._material_indices[material.name] = material.mid
        self._material_uids.add(material.uid)
        logger.debug("Registered new material:\n%s", material)
        return material.mid

    def register_pair(
        self, first: MaterialDescriptor, second: MaterialDescriptor, material_pair: MaterialPairProperties
    ):
        """
        Registers a new material pair with the manager.

        Args:
            first: The first material in the pair.
            second: The second material in the pair.
            material_pair: The properties of the material pair.

        Raises:
            ValueError: If either material is not already registered.
        """
        # Register the first material if it is not already registered
        if not self.has_material(first.name):
            self.register(first)

        # Register the second material if it is not already registered
        if not self.has_material(second.name):
            self.register(second)

        # Configure the material pair properties
        self.configure_pair(first=first.name, second=second.name, material_pair=material_pair)
        logger.debug("Registered new material pair: %s - %s", first.name, second.name)

    def configure_pair(self, first: int | str, second: int | str, material_pair: MaterialPairProperties):
        """
        Configures the properties of an existing material pair.

        Args:
            first: The index or name of the first material in the pair.
            second: The index or name of the second material in the pair.
            material_pair: The properties to set for the material pair.

        Raises:
            ValueError: If either material is not found.
        """
        # Get indices of the materials
        mid1 = self.index(first)
        mid2 = self.index(second)

        # Set the material pair properties
        self._pair_properties[material_pair_key(mid1, mid2)] = material_pair
        logger.debug("Configured material pair: %s - %s", self.materials[mid1].name, self.materials[mid2].name)

    def merge(self, other: MaterialManager):
        """
        Merges another MaterialManager into this one, combining their materials and material-pair properties.

        Args:
            other: The other MaterialManager to merge.

        Raises:
            ValueError: If there are conflicting material names or UIDs.
        """
        # Iterate over the materials in the other manager
        for mat in other.materials:
            if not self.has_material(mat.name):
                self.register(copy(mat))

        for key, pair_props in other._pair_properties.items():
            first = other.materials[key >> 32].name
            second = other.materials[key & 0xFFFFFFFF].name
            self.configure_pair(first=first, second=second, material_pair=pair_props)

    def __getitem__(self, key) -> MaterialDescriptor:
        """
        Retrieves a material descriptor by its index or name.

        Args:
            key: The name or index of the material.

        Returns:
            The material descriptor.

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
            return self.materials[self.index(key)]

    def index(self, key: str | int) -> int:
        """
        Retrieves the index of a material by its name or index.

        Args:
            key: The name or index of the material.

        Returns:
            The index of the material.

        Raises:
            ValueError: If the material is not found.
            TypeError: If the key is not a string or integer.
        """
        # Check if the name exists in the materials list
        if isinstance(key, str):
            if key in self._material_indices:
                return self._material_indices[key]
        elif isinstance(key, int):
            # If the name is an integer, return it directly if it is a valid index
            if 0 <= key < self.num_materials:
                return key
        else:
            raise TypeError("Name argument must be a string or integer.")

        # If not found, raise an error
        raise ValueError(f"Material with key '{key}' not found.")

    ###
    # Material Properties Data
    ###

    def restitution_vector(self) -> np.ndarray:
        """
        Generates a vector of restitution coefficients over all materials.

        Returns:
            A 1D numpy array containing per-material restitution coefficients.
        """
        # Get the number of materials
        num_materials = len(self._materials)

        # Initialize the restitution matrix
        restitution = np.full((num_materials,), -1, dtype=np.float32)

        # Fill the matrix with the restitution coefficients
        for i in range(num_materials):
            restitution[i] = self._materials[i].restitution

        # Return the restitution matrix as a numpy array
        return restitution

    def restitution_matrix(self) -> np.ndarray:
        """
        Generates a matrix of restitution coefficients for all material pairs.

        Returns:
            A flat NumPy array containing the lower triangle of the restitution matrix.
        """
        return self._pair_matrix("restitution")

    def static_friction_vector(self) -> np.ndarray:
        """
        Generates a vector of static friction coefficients over all materials.

        Returns:
            A 1D numpy array containing per-material static friction coefficients.
        """
        # Get the number of materials
        num_materials = len(self._materials)

        # Initialize the restitution matrix
        static_friction = np.full((num_materials,), -1, dtype=np.float32)

        # Fill the matrix with the restitution coefficients
        for i in range(num_materials):
            static_friction[i] = self._materials[i].static_friction

        # Return the restitution matrix as a numpy array
        return static_friction

    def static_friction_matrix(self) -> np.ndarray:
        """
        Generates a matrix of friction coefficients for all material pairs.

        Returns:
            A flat NumPy array containing the lower triangle of the static friction matrix.
        """
        return self._pair_matrix("static_friction")

    def dynamic_friction_vector(self) -> np.ndarray:
        """
        Generates a vector of dynamic friction coefficients over all materials.

        Returns:
            A 1D numpy array containing per-material dynamic friction coefficients.
        """
        # Get the number of materials
        num_materials = len(self._materials)

        # Initialize the restitution matrix
        dynamic_friction = np.full((num_materials,), -1, dtype=np.float32)

        # Fill the matrix with the restitution coefficients
        for i in range(num_materials):
            dynamic_friction[i] = self._materials[i].dynamic_friction

        # Return the restitution matrix as a numpy array
        return dynamic_friction

    def dynamic_friction_matrix(self) -> np.ndarray:
        """
        Generates a matrix of friction coefficients for all material pairs.

        Returns:
            A flat NumPy array containing the lower triangle of the dynamic friction matrix.
        """
        return self._pair_matrix("dynamic_friction")

    def _pair_matrix(self, property_name: str) -> np.ndarray:
        values = np.full(self.num_material_pairs, -1, dtype=np.float32)
        for key, properties in self._pair_properties.items():
            low, high = key >> 32, key & 0xFFFFFFFF
            values[high * (high + 1) // 2 + low] = getattr(properties, property_name)
        return values

    ###
    # Material Model Creation
    ###

    def make_materials_model(self) -> MaterialsModel:
        # Construct the per-material properties
        materials_rest = [self.restitution_vector()]
        materials_static_fric = [self.static_friction_vector()]
        materials_dynamic_fric = [self.dynamic_friction_vector()]
        return MaterialsModel(
            num_materials=self.num_materials,
            restitution=wp.array(materials_rest[0], dtype=wp.float32),
            static_friction=wp.array(materials_static_fric[0], dtype=wp.float32),
            dynamic_friction=wp.array(materials_dynamic_fric[0], dtype=wp.float32),
        )

    def make_material_pairs_model(self) -> MaterialPairsModel:
        keys = sorted(self._pair_properties)
        rows = [self._pair_properties[key] for key in keys]
        return MaterialPairsModel(
            num_material_pairs=len(keys),
            keys=wp.array(keys, dtype=wp.uint64),
            restitution=wp.array([row.restitution for row in rows], dtype=wp.float32),
            static_friction=wp.array([row.static_friction for row in rows], dtype=wp.float32),
            dynamic_friction=wp.array([row.dynamic_friction for row in rows], dtype=wp.float32),
        )
