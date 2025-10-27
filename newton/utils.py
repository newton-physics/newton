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

# ==================================================================================
# sim utils
# ==================================================================================
from ._src.sim import (
    color_graph,
    plot_graph,
)

__all__ = [
    "color_graph",
    "plot_graph",
]

# ==================================================================================
# mesh utils
# ==================================================================================
from ._src.utils.mesh import (
    create_box_mesh,
    create_capsule_mesh,
    create_cone_mesh,
    create_cylinder_mesh,
    create_plane_mesh,
    create_sphere_mesh,
)

__all__ += [
    "create_box_mesh",
    "create_capsule_mesh",
    "create_cone_mesh",
    "create_cylinder_mesh",
    "create_plane_mesh",
    "create_sphere_mesh",
]

# ==================================================================================
# spatial math
# TODO: move these to Warp?
# ==================================================================================
from ._src.core.spatial import (  # noqa: E402
    quat_between_axes,
    quat_decompose,
    quat_from_euler,
    quat_to_euler,
    quat_to_rpy,
    quat_twist,
    quat_twist_angle,
    transform_twist,
    transform_wrench,
    velocity_at_point,
)

__all__ += [
    "quat_between_axes",
    "quat_decompose",
    "quat_from_euler",
    "quat_to_euler",
    "quat_to_rpy",
    "quat_twist",
    "quat_twist_angle",
    "transform_twist",
    "transform_wrench",
    "velocity_at_point",
]

# ==================================================================================
# math utils
# TODO: move math utils to Warp?
# ==================================================================================
from ._src.utils import (  # noqa: E402
    boltzmann,
    compute_world_offsets,
    leaky_max,
    leaky_min,
    smooth_max,
    smooth_min,
    vec_abs,
    vec_allclose,
    vec_inside_limits,
    vec_leaky_max,
    vec_leaky_min,
    vec_max,
    vec_min,
)

__all__ += [
    "boltzmann",
    "compute_world_offsets",
    "leaky_max",
    "leaky_min",
    "smooth_max",
    "smooth_min",
    "vec_abs",
    "vec_allclose",
    "vec_inside_limits",
    "vec_leaky_max",
    "vec_leaky_min",
    "vec_max",
    "vec_min",
]

# ==================================================================================
# asset management
# ==================================================================================
from ._src.utils.download_assets import download_asset  # noqa: E402

__all__ += [
    "download_asset",
]

# ==================================================================================
# recorders
# ==================================================================================
from ._src.utils.recorder import RecorderBasic, RecorderModelAndState  # noqa: E402

__all__ += [
    "RecorderBasic",
    "RecorderModelAndState",
]

# ==================================================================================
# run benchmark
# ==================================================================================

from ._src.utils.benchmark import EventTracer, event_scope, run_benchmark  # noqa: E402

__all__ += [
    "EventTracer",
    "event_scope",
    "run_benchmark",
]
