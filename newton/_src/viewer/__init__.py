# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Viewer interface for Newton physics simulations.

This module provides a high-level, renderer-agnostic interface for interactive
visualization of Newton models and simulation states.

Example usage:
    ```python
    import newton
    from newton.viewer import ViewerGL

    # Create viewer with OpenGL backend
    viewer = ViewerGL(model)

    # Render simulation
    while viewer.is_running():
        viewer.begin_frame(time)
        viewer.log_state(state)
        viewer.log_points(particle_positions)
        viewer.end_frame()

    viewer.close()
    ```

Layers:
    The viewer supports rendering multiple models/solvers as overlays in a
    single window via the layer system. Call :meth:`ViewerBase.activate`
    to switch the "current write target"; every subsequent ``set_model`` /
    ``log_state`` / ``log_*`` call is routed into the active layer and
    object names are prefixed with ``/layers/<layer_id>`` so layers do not
    collide. Toggle visibility per layer via
    :meth:`ViewerBase.set_layer_visible` or the "Layers" group in the
    ``ViewerGL`` sidebar. See ``example_basic_multi_solver_overlay``.

    Picking, wind, and ``apply_forces`` are bound to the most recently
    activated layer's model.
"""

import importlib
from typing import TYPE_CHECKING

from .viewer import Layer, ViewerBase

if TYPE_CHECKING:
    from .picking import Picking
    from .viewer_file import ViewerFile
    from .viewer_gl import ViewerGL
    from .viewer_null import ViewerNull
    from .viewer_rerun import ViewerRerun
    from .viewer_rtx import ViewerRTX
    from .viewer_usd import ViewerUSD
    from .viewer_viser import ViewerViser

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "Picking": (".picking", "Picking"),
    "ViewerFile": (".viewer_file", "ViewerFile"),
    "ViewerGL": (".viewer_gl", "ViewerGL"),
    "ViewerNull": (".viewer_null", "ViewerNull"),
    "ViewerRerun": (".viewer_rerun", "ViewerRerun"),
    "ViewerRTX": (".viewer_rtx", "ViewerRTX"),
    "ViewerUSD": (".viewer_usd", "ViewerUSD"),
    "ViewerViser": (".viewer_viser", "ViewerViser"),
}

__all__ = [
    "Layer",
    "Picking",
    "ViewerBase",
    "ViewerFile",
    "ViewerGL",
    "ViewerNull",
    "ViewerRTX",
    "ViewerRerun",
    "ViewerUSD",
    "ViewerViser",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
