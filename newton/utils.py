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
from ._src.sim.graph_coloring import color_graph, plot_graph

__all__ = [
    "color_graph",
    "plot_graph",
]

# ==================================================================================
# mesh utils
# ==================================================================================
from ._src.utils.mesh import (
    MeshAdjacency,
    create_box_mesh,
    create_capsule_mesh,
    create_cone_mesh,
    create_cylinder_mesh,
    create_ellipsoid_mesh,
    create_plane_mesh,
    create_sphere_mesh,
    solidify_mesh,
)

__all__ += [
    "MeshAdjacency",
    "create_box_mesh",
    "create_capsule_mesh",
    "create_cone_mesh",
    "create_cylinder_mesh",
    "create_ellipsoid_mesh",
    "create_plane_mesh",
    "create_sphere_mesh",
    "solidify_mesh",
]

# ==================================================================================
# render utils
# ==================================================================================
from ._src.utils.render import (  # noqa: E402
    bourke_color_map,
)

__all__ += [
    "bourke_color_map",
]

# ==================================================================================
# cable utils
# ==================================================================================
from ._src.utils.cable import (  # noqa: E402
    create_cable_stiffness_from_elastic_moduli,
    create_parallel_transport_cable_quaternions,
    create_straight_cable_points,
    create_straight_cable_points_and_quaternions,
)

__all__ += [
    "create_cable_stiffness_from_elastic_moduli",
    "create_parallel_transport_cable_quaternions",
    "create_straight_cable_points",
    "create_straight_cable_points_and_quaternions",
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
from ._src.math import (  # noqa: E402
    boltzmann,
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
from ._src.utils import compute_world_offsets  # noqa: E402

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
# run benchmark
# ==================================================================================

from ._src.utils.benchmark import EventTracer, event_scope, run_benchmark  # noqa: E402

__all__ += [
    "EventTracer",
    "event_scope",
    "run_benchmark",
]

# ==================================================================================
# import utils
# ==================================================================================

from ._src.utils.import_utils import string_to_warp  # noqa: E402

__all__ += [
    "string_to_warp",
]
