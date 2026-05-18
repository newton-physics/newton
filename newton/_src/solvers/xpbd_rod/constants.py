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

"""Constants for XPBD Cosserat rod solver."""

from __future__ import annotations

# Warp tile configuration for direct solve
BLOCK_DIM = 128
TILE = 64

# Banded Cholesky layout
BAND_KD = 11
BAND_LDAB = 34

# Direct solve backend identifiers
DIRECT_SOLVE_BLOCK_THOMAS = "block_thomas"
DIRECT_SOLVE_SPLIT_THOMAS = "split_thomas"
DIRECT_SOLVE_BLOCK_JACOBI = "block_jacobi"
DIRECT_SOLVE_BANDED_CHOLESKY = "banded_cholesky"

DIRECT_SOLVE_BACKENDS = (
    DIRECT_SOLVE_BLOCK_THOMAS,
    DIRECT_SOLVE_SPLIT_THOMAS,
    DIRECT_SOLVE_BLOCK_JACOBI,
    DIRECT_SOLVE_BANDED_CHOLESKY,
)
