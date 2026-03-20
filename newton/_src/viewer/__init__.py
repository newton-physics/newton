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
"""

from .viewer_file import ViewerFile
from .viewer_gl import ViewerGL
from .viewer_null import ViewerNull
from .viewer_rerun import ViewerRerun
from .viewer_usd import ViewerUSD
from .viewer_viser import ViewerViser

__all__ = [
    "ViewerFile",
    "ViewerGL",
    "ViewerNull",
    "ViewerRerun",
    "ViewerUSD",
    "ViewerViser",
]
