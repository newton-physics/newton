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
