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

from typing import Any

import numpy as np
import warp as wp

from ..core.types import Axis
from ..math import (
    boltzmann,
    leaky_max,
    leaky_min,
    smooth_max,
    smooth_min,
    vec_abs,
    vec_allclose,
    vec_inside_limits,
    vec_leaky_max,
    vec_leaky_min,
    vec_max,
    vec_min,
)
from .download_assets import clear_git_cache, download_asset
from .texture import load_texture, normalize_texture
from .topology import topological_sort, topological_sort_undirected


def check_conditional_graph_support():
    """
    Check if conditional graph support is available in the current world.

    Returns:
        bool: True if conditional graph support is available, False otherwise.
    """
    return wp.is_conditional_graph_supported()


def compute_world_offsets(num_worlds: int, spacing: tuple[float, float, float], up_axis: Any = None):
    """
    Compute positional offsets for multiple worlds arranged in a grid.

    This function computes 3D offsets for arranging multiple worlds based on the provided spacing.
    The worlds are arranged in a regular grid pattern, with the layout automatically determined
    based on the non-zero dimensions in the spacing tuple.

    Args:
        num_worlds (int): The number of worlds to arrange.
        spacing (tuple[float, float, float]): The spacing between worlds along each axis.
            Non-zero values indicate active dimensions for the grid layout.
        up_axis (Any, optional): The up axis to ensure worlds are not shifted below the ground plane.
            If provided, the offset correction along this axis will be zero.

    Returns:
        np.ndarray: An array of shape (num_worlds, 3) containing the 3D offsets for each world.
    """
    # Handle edge case
    if num_worlds <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Compute positional offsets per world
    spacing = np.array(spacing, dtype=np.float32)
    nonzeros = np.nonzero(spacing)[0]
    num_dim = nonzeros.shape[0]

    if num_dim > 0:
        side_length = int(np.ceil(num_worlds ** (1.0 / num_dim)))
        spacings = []

        if num_dim == 1:
            for i in range(num_worlds):
                spacings.append(i * spacing)
        elif num_dim == 2:
            for i in range(num_worlds):
                d0 = i // side_length
                d1 = i % side_length
                offset = np.zeros(3)
                offset[nonzeros[0]] = d0 * spacing[nonzeros[0]]
                offset[nonzeros[1]] = d1 * spacing[nonzeros[1]]
                spacings.append(offset)
        elif num_dim == 3:
            for i in range(num_worlds):
                d0 = i // (side_length * side_length)
                d1 = (i // side_length) % side_length
                d2 = i % side_length
                offset = np.zeros(3)
                offset[0] = d0 * spacing[0]
                offset[1] = d1 * spacing[1]
                offset[2] = d2 * spacing[2]
                spacings.append(offset)

        spacings = np.array(spacings, dtype=np.float32)
    else:
        spacings = np.zeros((num_worlds, 3), dtype=np.float32)

    # Center the grid
    min_offsets = np.min(spacings, axis=0)
    correction = min_offsets + (np.max(spacings, axis=0) - min_offsets) / 2.0

    # Ensure the worlds are not shifted below the ground plane
    if up_axis is not None:
        correction[Axis.from_any(up_axis)] = 0.0

    spacings -= correction
    return spacings


__all__ = [
    "boltzmann",
    "check_conditional_graph_support",
    "clear_git_cache",
    "compute_world_offsets",
    "download_asset",
    "leaky_max",
    "leaky_min",
    "load_texture",
    "normalize_texture",
    "smooth_max",
    "smooth_min",
    "topological_sort",
    "topological_sort_undirected",
    "vec_abs",
    "vec_allclose",
    "vec_inside_limits",
    "vec_leaky_max",
    "vec_leaky_min",
    "vec_max",
    "vec_min",
]
