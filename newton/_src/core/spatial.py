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

from enum import Enum

import warp as wp


class SpatialVectorForm(Enum):
    """Component ordering in 6D spatial vectors."""

    LINEAR_ANGULAR = "linear_angular"  # [fx, fy, fz, τx, τy, τz] - Newton/Warp, ROS
    ANGULAR_LINEAR = "angular_linear"  # [τx, τy, τz, fx, fy, fz] - MuJoCo, Featherstone


@wp.func
def swap_spatial_halves(x: wp.spatial_vector) -> wp.spatial_vector:
    """Swap [a, b] ↔ [b, a] in a spatial vector.

    Converts between component orderings:
    - [linear, angular] ↔ [angular, linear]
    - Newton/Warp [f, τ] ↔ MuJoCo [τ, f]

    This is a symmetric operation (its own inverse):
    swap_spatial_halves(swap_spatial_halves(x)) = x

    Args:
        x: Spatial vector in either convention

    Returns:
        Spatial vector with halves swapped

    Example:
        # MuJoCo cfrc_int [τ, f] → Newton/Warp [f, τ]
        newton_wrench = swap_spatial_halves(mujoco_wrench)
    """
    return wp.spatial_vector(wp.spatial_bottom(x), wp.spatial_top(x))
