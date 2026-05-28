# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides utility functions for model
builder composition and manipulation.

This module includes functions to add common
modifiers to model builders, such as ground
planes, as well as factory functions to create
homogeneous multi-world builders and import
USD models.
"""

import os
import time
from collections.abc import Callable

import warp as wp

from ...core.builder import ModelBuilderKamino
from ...core.shapes import BoxShape, PlaneShape
from ...core.types import transformf, vec3f, vec6f
from ...utils.io.usd import USDImporter

###
# Module interface
###

__all__ = [
    "add_ground_box",
    "add_ground_plane",
    "build_usd",
    "make_homogeneous_builder",
    "set_uniform_body_pose_offset",
    "set_uniform_body_twist_offset",
]


###
# Modifiers
###


def add_ground_plane(
    builder: ModelBuilderKamino,
    group: int = 1,
    collides: int = 1,
    world_index: int = 0,
    z_offset: float = 0.0,
) -> int:
    """
    Adds a static plane geometry to a given builder to represent a flat ground with infinite dimensions.

    Args:
        builder (ModelBuilderKamino):
            The model builder to which the ground plane should be added.
        group (int):
            The collision group for the ground geometry.\n
            Defaults to `1`.
        collides (int):
            The collision mask for the ground geometry.\n
            Defaults to `1`.
        world_index (int):
            The index of the world in the builder where the ground geometry should be added.\n
            If the value does not correspond to an existing world an error will be raised.\n
            Defaults to `0`.
        z_offset (float):
            The vertical offset of the ground plane along the Z axis.\n
            Defaults to `0.0`.
    Returns:
        int: The ID of the added ground geometry.
    """
    return builder.add_geometry(
        shape=PlaneShape(vec3f(0.0, 0.0, 1.0), 0.0),
        offset=transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        name="ground",
        group=group,
        collides=collides,
        world_index=world_index,
    )


def add_ground_box(
    builder: ModelBuilderKamino,
    group: int = 1,
    collides: int = 1,
    world_index: int = 0,
    z_offset: float = 0.0,
) -> int:
    """
    Adds a static box geometry to a given builder to represent a flat ground with finite dimensions.

    Args:
        builder (ModelBuilderKamino):
            The model builder to which the ground box should be added.
        group (int):
            The collision group for the ground geometry.\n
            Defaults to `1`.
        collides (int):
            The collision mask for the ground geometry.\n
            Defaults to `1`.
        world_index (int):
            The index of the world in the builder where the ground geometry should be added.\n
            If the value does not correspond to an existing world an error will be raised.\n
            Defaults to `0`.
        z_offset (float):
            The vertical offset of the ground box along the Z axis.\n
            Defaults to `0.0`.

    Returns:
        int: The ID of the added ground geometry.
    """
    return builder.add_geometry(
        shape=BoxShape(10.0, 10.0, 0.5),
        offset=transformf(0.0, 0.0, -0.5 + z_offset, 0.0, 0.0, 0.0, 1.0),
        name="ground",
        group=group,
        collides=collides,
        world_index=world_index,
    )


def set_uniform_body_pose_offset(builder: ModelBuilderKamino, offset: transformf):
    """
    Offsets the initial poses of all rigid bodies existing in the builder uniformly by the specified offset.

    Args:
        builder (ModelBuilderKamino): The model builder containing the bodies to offset.
        offset (transformf): The pose offset to apply to each body in the builder in the form of a :class:`transformf`.
    """
    for body in builder.all_bodies:
        body.q_i_0 = wp.mul(offset, body.q_i_0)


def set_uniform_body_twist_offset(builder: ModelBuilderKamino, offset: vec6f):
    """
    Offsets the initial twists of all rigid bodies existing in the builder uniformly by the specified offset.

    Args:
        builder (ModelBuilderKamino): The model builder containing the bodies to offset.
        offset (vec6f): The twist offset to apply to each body in the builder in the form of a :class:`vec6f`.
    """
    for body in builder.all_bodies:
        body.u_i_0 += offset


###
# Builder utilities
###


def build_usd(
    source: str,
    load_drive_dynamics: bool = True,
    load_static_geometry: bool = True,
    ground: bool = True,
) -> ModelBuilderKamino:
    """
    Imports a USD model and optionally adds a ground plane.

    Each call creates a new world with the USD model and optional ground plane.

    Args:
        source: Path to USD file
        load_drive_dynamics: Whether to load drive parameters from USD. Necessary for using implicit PD
        load_static_geometry: Whether to load static geometry from USD
        ground: Whether to add a ground plane

    Returns:
        ModelBuilderKamino with imported USD model and optional ground plane
    """
    # Import the USD model
    importer = USDImporter()
    _builder = importer.import_from(
        source=source,
        load_drive_dynamics=load_drive_dynamics,
        load_static_geometry=load_static_geometry,
    )

    # Optionally add ground geometry
    if ground:
        add_ground_box(builder=_builder, group=1, collides=1)

    # Return the builder constructed from the USD model
    return _builder


def make_homogeneous_builder(num_worlds: int, build_fn: Callable, show_progress=False, **kwargs) -> ModelBuilderKamino:
    """
    Utility factory function to create a multi-world builder with identical worlds replicated across the model.

    Args:
        num_worlds (int): The number of worlds to create.
        build_fn (callable): The model builder function to use.
        show_progress (bool): Whether to display a progress bar as the worlds are being replicated.
        **kwargs: Additional keyword arguments to pass to the builder function.

    Returns:
        ModelBuilderKamino: The constructed model builder.
    """
    # First build a single world
    # NOTE: We want to do this first to avoid re-constructing the same model multiple
    # times especially if the construction is expensive such as importing from USD.
    single = build_fn(**kwargs)

    # Then replicate it across the specified number of worlds
    builder = ModelBuilderKamino(default_world=False)
    start_time = time.time()
    for i in range(num_worlds):
        if show_progress:
            from ....examples import print_progress_bar  # noqa: PLC0415

            print_progress_bar(i + 1, num_worlds, start_time, prefix="Adding builders", suffix="")
        builder.add_builder(single)
    return builder


###
# Asset path utilities
###


def get_basics_usd_assets_path() -> str:
    """
    Returns the path to the USD assets for basic models.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../assets/basics")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for basic models does not exist: {path}")
    return path


def get_testing_usd_assets_path() -> str:
    """
    Returns the path to the USD assets for testing models.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../assets/testing")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for testing models does not exist: {path}")
    return path
