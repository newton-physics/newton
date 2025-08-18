# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Simplified viewer interface for Newton physics simulations.

This module provides a high-level, renderer-agnostic interface for interactive
visualization of Newton models and simulation states. The design follows the
dependency injection pattern where the Viewer class takes a renderer backend
implementing the RendererBase interface.

Example usage:
    ```python
    import newton
    from newton.viewer import ViewerGL

    # Create viewer with OpenGL backend
    viewer = ViewerGL(model)

    # Render simulation
    while viewer.is_running():
        viewer.begin_frame(time)
        viewer.log_model(state)
        viewer.log_points(particle_positions)
        viewer.end_frame()

    viewer.close()
    ```
"""

from .viewer_gl import ViewerGL
from .viewer_rerun import ViewerRerun
from .viewer_usd import ViewerUSD

__all__ = [
    "ViewerGL",
    "ViewerRerun",
    "ViewerUSD",
]
