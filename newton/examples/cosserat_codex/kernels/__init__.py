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

"""Warp kernels for Cosserat rod simulation."""

from __future__ import annotations

# Re-export all kernels for convenience
from .assembly import (
    _warp_assemble_jmjt_banded,
    _warp_assemble_jmjt_blocks,
    _warp_assemble_jmjt_dense,
    _warp_pad_diagonal,
)
from .collision import (
    _warp_apply_concentric_constraint,
    _warp_apply_direct_corrections,
    _warp_apply_floor_collisions,
    _warp_apply_root_translation,
    _warp_apply_track_sliding,
    _warp_build_segment_lines,
    _warp_compute_inv_inertia_world,
    _warp_constraint_max,
    _warp_copy_from_offset,
    _warp_copy_with_offset,
    _warp_set_root_on_track,
    _warp_set_root_orientation,
    _warp_update_velocities_from_positions,
    _warp_zero_2d,
    _warp_zero_float,
    _warp_zero_root_velocities,
)
from .constraints import (
    _warp_build_rhs,
    _warp_compute_jacobians_direct,
    _warp_prepare_compliance,
    _warp_update_constraints_direct,
)
from .integration import (
    _warp_integrate_positions,
    _warp_integrate_rotations,
    _warp_predict_positions,
    _warp_predict_rotations,
)
from .math import (
    _block_column,
    _block_index,
    _block_mul,
    _block_mul_vec,
    _block_row,
    _block_set_column,
    _block_solve,
    _block_sub,
    _inv_inertia_mul_vec,
    _load_block,
    _load_vec,
    _mat33_add,
    _mat33_cholesky,
    _mat33_inverse,
    _mat33_mul,
    _mat33_mul_vec3,
    _mat33_solve_lower,
    _mat33_solve_upper,
    _mat33_sub,
    _mat33_transpose,
    _store_block,
    _store_vec,
    _vec_index,
    _warp_jacobian_index,
    _warp_quat_conjugate,
    _warp_quat_mul,
    _warp_quat_normalize,
    _warp_quat_rotate_vector,
)
from .solvers import (
    _warp_block_thomas_solve,
    _warp_cholesky_solve_tile,
    _warp_spbsv_u11_1rhs,
    _warp_spbsv_u11_1rhs_iter_ref,
)

__all__ = [
    # Math functions
    "_warp_quat_mul",
    "_warp_quat_conjugate",
    "_warp_quat_normalize",
    "_warp_quat_rotate_vector",
    "_warp_jacobian_index",
    "_mat33_add",
    "_mat33_sub",
    "_mat33_mul",
    "_mat33_mul_vec3",
    "_mat33_transpose",
    "_mat33_cholesky",
    "_mat33_solve_lower",
    "_mat33_solve_upper",
    "_mat33_inverse",
    "_block_index",
    "_vec_index",
    "_load_block",
    "_store_block",
    "_load_vec",
    "_store_vec",
    "_block_column",
    "_block_set_column",
    "_block_row",
    "_block_mul",
    "_block_sub",
    "_block_mul_vec",
    "_block_solve",
    "_inv_inertia_mul_vec",
    # Integration kernels
    "_warp_predict_positions",
    "_warp_integrate_positions",
    "_warp_predict_rotations",
    "_warp_integrate_rotations",
    # Constraint kernels
    "_warp_prepare_compliance",
    "_warp_update_constraints_direct",
    "_warp_compute_jacobians_direct",
    "_warp_build_rhs",
    # Assembly kernels
    "_warp_assemble_jmjt_dense",
    "_warp_assemble_jmjt_banded",
    "_warp_assemble_jmjt_blocks",
    "_warp_pad_diagonal",
    # Solver kernels
    "_warp_cholesky_solve_tile",
    "_warp_block_thomas_solve",
    "_warp_spbsv_u11_1rhs",
    "_warp_spbsv_u11_1rhs_iter_ref",
    # Collision kernels
    "_warp_apply_floor_collisions",
    "_warp_apply_direct_corrections",
    "_warp_constraint_max",
    "_warp_zero_float",
    "_warp_zero_2d",
    "_warp_copy_with_offset",
    "_warp_copy_from_offset",
    "_warp_build_segment_lines",
    "_warp_apply_root_translation",
    "_warp_zero_root_velocities",
    "_warp_set_root_orientation",
    "_warp_update_velocities_from_positions",
    "_warp_apply_track_sliding",
    "_warp_set_root_on_track",
    "_warp_apply_concentric_constraint",
    "_warp_compute_inv_inertia_world",
]
