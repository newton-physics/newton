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
Style3D solver module.

This module provides the :class:`~newton.solvers.SolverStyle3D` cloth simulation
solver along with helper functions for setting up cloth assets. It includes
utilities for creating cloth meshes and grids, handling collisions, and sewing
cloth vertices together.
"""

from .cloth import (
    add_cloth_grid,
    add_cloth_mesh,
)
from .solver_style3d import SolverStyle3D

__all__ = [
    "SolverStyle3D",
    "add_cloth_grid",
    "add_cloth_mesh",
]
