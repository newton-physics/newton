# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Spatial vector convention conversion utilities."""

import numpy as np
import warp as wp

from newton._src.core.spatial import SpatialVectorForm


def convert_spatial_vector_batch(
    data: np.ndarray | wp.array,
    from_form: SpatialVectorForm,
    to_form: SpatialVectorForm,
) -> np.ndarray:
    """Convert spatial vector batch between conventions.

    Handles only component ordering (form), not frame transformations.
    For frame conversion, use transform_wrench() or transform_twist().

    Args:
        data: Spatial vectors [..., 6]
        from_form: Source ordering
        to_form: Target ordering

    Returns:
        Converted spatial vectors (NumPy array)

    Example:
        # Convert MuJoCo [angular, linear] to Newton [linear, angular]
        newton_wrenches = convert_spatial_vector_batch(
            mujoco_wrenches,
            from_form=SpatialVectorForm.ANGULAR_LINEAR,
            to_form=SpatialVectorForm.LINEAR_ANGULAR
        )
    """
    if from_form == to_form:
        return data if isinstance(data, np.ndarray) else data.numpy()

    # Convert warp array to numpy
    np_data = data if isinstance(data, np.ndarray) else data.numpy()

    # Swap halves
    result = np.empty_like(np_data)
    result[..., :3] = np_data[..., 3:]
    result[..., 3:] = np_data[..., :3]
    return result
