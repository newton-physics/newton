# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING

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
from ._src.geometry.utils import remesh_mesh
from ._src.utils.mesh import (
    MeshAdjacency,
    MeshAdjacencyData,
    solidify_mesh,
    validate_tet_mesh,
    validate_triangle_mesh,
)

__all__ += [
    "MeshAdjacency",
    "MeshAdjacencyData",
    "remesh_mesh",
    "solidify_mesh",
    "validate_tet_mesh",
    "validate_triangle_mesh",
]

from ._src.utils.heightfield import rasterize_mesh_to_heightfield  # noqa: E402

__all__ += [
    "rasterize_mesh_to_heightfield",
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
# color utils
# ==================================================================================

from ._src.utils.color import (  # noqa: E402
    ColorSpace,
    color_linear_to_srgb,
    color_srgb_to_linear,
)

__all__ += [
    "ColorSpace",
    "color_linear_to_srgb",
    "color_srgb_to_linear",
]

# ==================================================================================
# cable and rod utils
# ==================================================================================
from ._src.utils.cable import (  # noqa: E402
    RodStiffness,
    cable_straight_points,
    create_cable_stiffness_from_elastic_moduli,
    create_parallel_transport_cable_quaternions,
    create_straight_cable_points,
    create_straight_cable_points_and_quaternions,
    rod_parallel_transport_quaternions,
    rod_stiffness_from_elastic_moduli,
    rod_straight_points_and_quaternions,
)

if TYPE_CHECKING:
    CableStiffness = RodStiffness

__all__ += [
    "CableStiffness",
    "RodStiffness",
    "cable_straight_points",
    "create_cable_stiffness_from_elastic_moduli",
    "create_parallel_transport_cable_quaternions",
    "create_straight_cable_points",
    "create_straight_cable_points_and_quaternions",
    "rod_parallel_transport_quaternions",
    "rod_stiffness_from_elastic_moduli",
    "rod_straight_points_and_quaternions",
]

_DEPRECATED_CABLE_SYMBOLS = {
    "CableStiffness": RodStiffness,
}

__deprecated_symbols__ = {
    "CableStiffness": "Deprecated in 1.6; use RodStiffness instead.",
    "create_cable_stiffness_from_elastic_moduli": ("Deprecated in 1.6; use rod_stiffness_from_elastic_moduli instead."),
    "create_parallel_transport_cable_quaternions": (
        "Deprecated in 1.6; use rod_parallel_transport_quaternions instead."
    ),
    "create_straight_cable_points": "Deprecated in 1.6; use cable_straight_points instead.",
    "create_straight_cable_points_and_quaternions": (
        "Deprecated in 1.6; use rod_straight_points_and_quaternions instead."
    ),
}


def __getattr__(name: str):
    try:
        value = _DEPRECATED_CABLE_SYMBOLS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    warnings.warn(
        f"newton.utils.{name} is deprecated in Newton 1.6; use newton.utils.RodStiffness instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEPRECATED_CABLE_SYMBOLS))


# ==================================================================================
# world utils
# ==================================================================================
from ._src.utils import compute_world_offsets  # noqa: E402

__all__ += [
    "compute_world_offsets",
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

# ==================================================================================
# texture utils
# ==================================================================================

from ._src.utils.texture import load_texture, normalize_texture  # noqa: E402

__all__ += [
    "load_texture",
    "normalize_texture",
]
