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

from .viewer import Layer, ViewerBase
from .viewer_file import ViewerFile
from .viewer_gl import ViewerGL
from .viewer_null import ViewerNull
from .viewer_rerun import ViewerRerun
from .viewer_rtx import ViewerRTX
from .viewer_usd import ViewerUSD
from .viewer_viser import ViewerViser

__all__ = [
    "Layer",
    "ViewerBase",
    "ViewerFile",
    "ViewerGL",
    "ViewerNull",
    "ViewerRTX",
    "ViewerRerun",
    "ViewerUSD",
    "ViewerViser",
]
