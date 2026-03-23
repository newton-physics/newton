# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Linear Algebra: Matrix factorization implementations (kernels and launchers)"""

from .llt_blocked import (
    llt_blocked_factorize,
    llt_blocked_solve,
    llt_blocked_solve_inplace,
    make_llt_blocked_factorize_kernel,
    make_llt_blocked_solve_inplace_kernel,
    make_llt_blocked_solve_kernel,
)
from .llt_sequential import (
    _llt_sequential_factorize,
    _llt_sequential_solve,
    _llt_sequential_solve_inplace,
    llt_sequential_factorize,
    llt_sequential_solve,
    llt_sequential_solve_inplace,
)

###
# Module API
###

__all__ = [
    "_llt_sequential_factorize",
    "_llt_sequential_solve",
    "_llt_sequential_solve_inplace",
    "llt_blocked_factorize",
    "llt_blocked_solve",
    "llt_blocked_solve_inplace",
    "llt_sequential_factorize",
    "llt_sequential_solve",
    "llt_sequential_solve_inplace",
    "make_llt_blocked_factorize_kernel",
    "make_llt_blocked_solve_inplace_kernel",
    "make_llt_blocked_solve_kernel",
]
