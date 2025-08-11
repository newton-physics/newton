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

__all__ = []

try:
    from .viewer_gl import ViewerGL  # noqa: F401

    __all__.append("ViewerGL")
except ImportError:
    # OpenGL not available
    pass

try:
    from .viewer_rerun import ViewerRerun  # noqa: F401

    __all__.append("ViewerRerun")
except ImportError:
    # Rerun not available
    pass

try:
    from .viewer_usd import ViewerUSD  # noqa: F401

    __all__.append("ViewerUSD")
except ImportError:
    # USD (pxr) not available
    pass
