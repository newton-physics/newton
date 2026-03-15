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
XPBD Rod solver module.

This module provides :func:`add_elastic_rod` for setting up Cosserat elastic
rod assets. Use :class:`~newton.solvers.SolverXPBDRod` as the canonical public
solver import path.
"""

from .builder import add_elastic_rod
from .kernels_visualization import compute_director_lines_kernel
from .solver_xpbd_rod import SolverXPBDRod

__all__ = [
    "SolverXPBDRod",
    "add_elastic_rod",
    "compute_director_lines_kernel",
]
