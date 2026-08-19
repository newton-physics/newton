# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Linear Algebra: Type parameters for generic linalg classes.

Provides shared :class:`TypeVar` symbols used to statically parameterize
the linalg classes (``DenseSquareMultiLinearInfo``, ``BlockSparseMatrices``,
``LinearSolver``, ...) so that ``wp.array`` annotations carry precise dtypes.

Runtime behavior is unaffected: the existing ``FloatType`` / ``IntType`` /
``VecIntType`` unions in :mod:`newton._src.solvers.kamino._src.core.types`
remain the source of truth for ``issubclass`` / ``get_args`` checks; these
TypeVars live alongside them as static-only siblings.
"""

from typing import TypeVar

import warp as wp

__all__ = [
    "BlockScalarType",
    "BlockType",
    "IndexType",
    "ScalarType",
]


ScalarType = TypeVar("ScalarType", wp.float16, wp.float32, wp.float64)
"""Float-only scalar element type. Mirrors ``FloatType``."""

BlockScalarType = TypeVar(
    "BlockScalarType",
    wp.float16,
    wp.float32,
    wp.float64,
    wp.int16,
    wp.int32,
    wp.int64,
)
"""Float or integer scalar element type. Mirrors ``FloatType | IntType``"""

IndexType = TypeVar("IndexType", wp.int16, wp.int32, wp.int64)
"""Integer type used for index types. Mirrors ``IntType``."""

BlockType = TypeVar("BlockType")
"""
Warp dtype of a non-zero block in a block-sparse matrix.

Unconstrained because the concrete block class is built dynamically at runtime.
"""
