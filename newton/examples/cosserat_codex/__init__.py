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

"""Cosserat rod simulation example integrating DefKitAdv with Newton.

This package provides a modular implementation of Cosserat elastic rod
simulation using XPBD (eXtended Position-Based Dynamics) with GPU
acceleration via NVIDIA Warp.

Package Structure:
    constants: Shared constants for simulation parameters.
    math_utils: Host-side math utilities (quaternions, etc.).
    kernels/: Warp GPU kernels for simulation steps.
    rod/: Rod state implementations (DLL, NumPy, Warp GPU).
    solver/: XPBD constraint solver.
    simulation/: High-level simulation and example classes.
"""

from __future__ import annotations

# Re-export core classes for backward compatibility
from .constants import (
    BAND_KD,
    BAND_LDAB,
    BLOCK_DIM,
    DIRECT_SOLVE_BACKENDS,
    DIRECT_SOLVE_CPU_NUMPY,
    DIRECT_SOLVE_WARP_BANDED_CHOLESKY,
    DIRECT_SOLVE_WARP_BLOCK_THOMAS,
    TILE,
)
from .math_utils import quat_from_axis_angle, quat_multiply, rotate_vector_by_quaternion
from .rod import (
    BtQuaternion,
    BtVector3,
    DefKitDirectLibrary,
    DefKitDirectRodState,
    NumpyDirectRodState,
    RodBatch,
    RodConfig,
    RodState,
    RodStateBase,
    WarpResidentRodState,
)
from .solver import CosseratXPBDSolver

__all__ = [
    # Constants
    "BAND_KD",
    "BAND_LDAB",
    "BLOCK_DIM",
    "DIRECT_SOLVE_BACKENDS",
    "DIRECT_SOLVE_CPU_NUMPY",
    "DIRECT_SOLVE_WARP_BANDED_CHOLESKY",
    "DIRECT_SOLVE_WARP_BLOCK_THOMAS",
    "TILE",
    # Math utilities
    "quat_from_axis_angle",
    "quat_multiply",
    "rotate_vector_by_quaternion",
    # Rod classes
    "BtQuaternion",
    "BtVector3",
    "DefKitDirectLibrary",
    "DefKitDirectRodState",
    "NumpyDirectRodState",
    "RodBatch",
    "RodConfig",
    "RodState",
    "RodStateBase",
    "WarpResidentRodState",
    # Solver
    "CosseratXPBDSolver",
]
