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

"""Linear Algebra Utilities"""

from .matrix import (
    MatrixComparison,
    MatrixSign,
    RectangularMatrixProperties,
    SquareSymmetricMatrixProperties,
    is_square_matrix,
    is_symmetric_matrix,
)
from .rand import (
    eigenvalues_from_distribution,
    random_rhs_for_matrix,
    random_spd_matrix,
    random_symmetric_matrix,
)
from .range import (
    in_range_via_gaussian_elimination,
    in_range_via_left_nullspace,
    in_range_via_projection,
    in_range_via_rank,
    in_range_via_residual,
)

###
# Module interface
###

__all__ = [
    "MatrixComparison",
    "MatrixSign",
    "RectangularMatrixProperties",
    "SquareSymmetricMatrixProperties",
    "eigenvalues_from_distribution",
    "in_range_via_gaussian_elimination",
    "in_range_via_left_nullspace",
    "in_range_via_lu",
    "in_range_via_projection",
    "in_range_via_rank",
    "in_range_via_residual",
    "is_square_matrix",
    "is_symmetric_matrix",
    "random_rhs_for_matrix",
    "random_spd_matrix",
    "random_symmetric_matrix",
]
