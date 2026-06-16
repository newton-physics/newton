# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

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
    MeshAdjacency as _SoftMeshAdjacency,
)
from ._src.utils.mesh import (
    solidify_mesh,
    validate_tet_mesh,
    validate_triangle_mesh,
)


class MeshAdjacency:
    """Deprecated triangle-mesh edge adjacency helper.

    Use :attr:`newton.Model.soft_mesh_adjacency` for simulation adjacency data.

    This thin compatibility wrapper reproduces the legacy ``.edges`` dict by
    delegating edge extraction to the refactored soft-mesh adjacency in
    :mod:`newton._src.utils.mesh`, so a single edge-walking implementation backs
    both the deprecated public helper and the internal simulation path.
    """

    @dataclass(slots=True)
    class Edge:
        v0: int
        v1: int
        o0: int
        o1: int
        f0: int
        f1: int

    def __init__(self, indices: Sequence[Sequence[int]] | np.ndarray):
        warnings.warn(
            "newton.utils.MeshAdjacency is deprecated; use Model.soft_mesh_adjacency for simulation data.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.indices = indices
        edge_indices, edge_tri_indices, _ = _SoftMeshAdjacency.compute_edge_adjacency(indices)
        self.edges: dict[tuple[int, int], MeshAdjacency.Edge] = {
            (min(int(v0), int(v1)), max(int(v0), int(v1))): MeshAdjacency.Edge(
                int(v0), int(v1), int(o0), int(o1), int(f0), int(f1)
            )
            for (o0, o1, v0, v1), (f0, f1) in zip(edge_indices, edge_tri_indices, strict=True)
        }


__all__ += [
    "MeshAdjacency",
    "remesh_mesh",
    "solidify_mesh",
    "validate_tet_mesh",
    "validate_triangle_mesh",
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
