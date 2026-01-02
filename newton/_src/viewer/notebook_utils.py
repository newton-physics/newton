# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for displaying Viser viewers in Jupyter notebooks and Sphinx docs.

This module provides functions to automatically detect the execution environment
and display either the interactive Viser viewer (in Jupyter) or an embedded
iframe player (in Sphinx-built documentation).
"""

from pathlib import Path


def is_sphinx_build() -> bool:
    """
    Detect if we're running inside a Sphinx documentation build (via nbsphinx).

    Returns:
        True if running in Sphinx/nbsphinx, False if in regular Jupyter session.
    """
    import os

    # nbsphinx sets SPHINXBUILD or we can check for sphinx in the call stack
    if os.environ.get("SPHINXBUILD"):
        return True

    # Check if sphinx is in the module list (imported during doc build)
    import sys

    if "sphinx" in sys.modules or "nbsphinx" in sys.modules:
        return True

    # Check call stack for sphinx-related frames
    try:
        import traceback

        for frame_info in traceback.extract_stack():
            if "sphinx" in frame_info.filename.lower() or "nbsphinx" in frame_info.filename.lower():
                return True
    except Exception:
        pass

    return False


def display_viser_viewer(
    viewer,
    recording_path: str | Path | None = None,
    height: int = 600,
    save_recording: bool = True,
):
    """
    Display a Viser viewer appropriately based on the execution environment.

    In a regular Jupyter notebook session, this displays the interactive Viser viewer.
    In a Sphinx documentation build, this displays an embedded iframe player that
    loads a pre-recorded .viser file.

    Args:
        viewer: The ViewerViser instance to display.
        recording_path: Path to save/load the .viser recording file.
            If None, a default path based on the notebook name is used.
        height: Height of the embedded player in pixels (for Sphinx builds).
        save_recording: If True and recording is active, save it before displaying.

    Returns:
        The display object (viewer for Jupyter, HTML for Sphinx).

    Example:
        >>> viewer = newton.viewer.ViewerViser()
        >>> viewer.set_model(model)
        >>> viewer.start_recording(fps=60)
        >>> # ... run simulation ...
        >>> display_viser_viewer(viewer, "_static/recordings/my_sim.viser")
    """
    from IPython.display import HTML, display

    if is_sphinx_build():
        # Sphinx build - save recording and display iframe
        if save_recording and recording_path and hasattr(viewer, "save_recording"):
            recording_path = Path(recording_path)
            recording_path.parent.mkdir(parents=True, exist_ok=True)
            viewer.save_recording(str(recording_path))
            print(f"Recording saved to: {recording_path}")

        # Generate iframe HTML that uses the shared viser-player.js
        # The data-recording path should be relative to the docs root
        if recording_path:
            # Convert to a path relative to _static if needed
            recording_str = str(recording_path)
            if not recording_str.startswith("_static"):
                # Assume it's a relative path from the notebook location
                # For Sphinx, we need to reference from the page location
                recording_str = recording_str.lstrip("./").lstrip("../")

        embed_html = f"""
<div class="viser-player-container" style="margin: 20px 0;">
    <p><strong>Interactive 3D Simulation Playback:</strong></p>
    <iframe
        class="viser-player"
        data-recording="{recording_str}"
        width="100%"
        height="{height}"
        frameborder="0"
        style="border: 1px solid #ccc; border-radius: 8px;">
    </iframe>
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">
        Use mouse to rotate, scroll to zoom, and right-click to pan.
    </p>
</div>
<script src="../_static/viser-player.js"></script>
"""
        return display(HTML(embed_html))
    else:
        # Regular Jupyter session - display the interactive viewer
        if save_recording and recording_path and hasattr(viewer, "save_recording"):
            # Still save the recording for future doc builds
            recording_path = Path(recording_path)
            recording_path.parent.mkdir(parents=True, exist_ok=True)
            viewer.save_recording(str(recording_path))
            print(f"Recording saved to: {recording_path}")

        # Display the interactive viewer
        return display(viewer)


def create_viser_embed_html(recording_path: str, height: int = 600) -> str:
    """
    Create HTML for embedding a Viser recording player.

    This is useful for manually creating embed code or for use in Sphinx directives.

    Args:
        recording_path: Path to the .viser recording file (relative to _static).
        height: Height of the player in pixels.

    Returns:
        HTML string with the embedded player.
    """
    return f"""
<div class="viser-player-container" style="margin: 20px 0;">
    <iframe
        class="viser-player"
        data-recording="{recording_path}"
        width="100%"
        height="{height}"
        frameborder="0"
        style="border: 1px solid #ccc; border-radius: 8px;">
    </iframe>
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">
        Use mouse to rotate, scroll to zoom, and right-click to pan.
    </p>
</div>
"""


def display_viser_recording(recording_path: str | Path, width: int = 800, height: int = 600):
    """
    Display a viser recording file with timeline controls in a Jupyter notebook.

    This uses viser's built-in client with playback mode, which provides:
    - Play/pause button
    - Timeline slider for seeking
    - Playback speed controls (0.5x, 1x, 2x, 4x, 8x)
    - Keyboard shortcuts (spacebar for play/pause)

    Args:
        recording_path: Path to the .viser recording file.
        width: Width of the player in pixels.
        height: Height of the player in pixels.

    Returns:
        IPython display object with the embedded player.

    Example:
        >>> viewer = ViewerViser(record_to_viser="simulation.viser")
        >>> # ... run simulation ...
        >>> viewer.show_notebook()  # Automatically uses this function
    """
    import socket
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    from IPython.display import IFrame, display

    recording_path = Path(recording_path).resolve()
    if not recording_path.exists():
        raise FileNotFoundError(f"Recording file not found: {recording_path}")

    # Get viser client directory from docs/_static/viser
    # This is a curated version of the viser client for notebook playback
    import newton

    newton_root = Path(newton.__file__).parent.parent
    viser_client_dir = newton_root / "docs" / "_static" / "viser"

    if not viser_client_dir.exists():
        raise FileNotFoundError(
            f"Viser client not found at {viser_client_dir}. "
            "Please ensure docs/_static/viser contains the viser client files."
        )

    # Read the recording file content
    recording_bytes = recording_path.read_bytes()

    # Find an available port
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    port = find_free_port()

    # Create a custom HTTP handler factory that serves both viser client and the recording
    def make_handler(recording_data: bytes, client_dir: str):
        class RecordingHandler(SimpleHTTPRequestHandler):
            # Fix MIME types for JavaScript and other files (Windows often has wrong mappings)
            extensions_map = {
                ".html": "text/html",
                ".htm": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
                ".mjs": "application/javascript",
                ".json": "application/json",
                ".wasm": "application/wasm",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".ico": "image/x-icon",
                ".ttf": "font/ttf",
                ".woff": "font/woff",
                ".woff2": "font/woff2",
                ".hdr": "application/octet-stream",
                ".viser": "application/octet-stream",
                "": "application/octet-stream",
            }

            def __init__(self, *args, **kwargs):
                self.recording_data = recording_data
                super().__init__(*args, directory=client_dir, **kwargs)

            def do_GET(self):
                # Parse path without query string
                path = self.path.split("?")[0]

                # Serve the recording file at /recording.viser
                if path == "/recording.viser":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(self.recording_data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(self.recording_data)
                else:
                    # Serve viser client files
                    super().do_GET()

            def log_message(self, format, *args):
                pass  # Suppress log messages

        return RecordingHandler

    handler_class = make_handler(recording_bytes, str(viser_client_dir))
    # Bind to all interfaces so IFrame can access it
    server = HTTPServer(("", port), handler_class)

    # Start server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Give the server a moment to start and verify it's running
    import time
    import urllib.request

    time.sleep(0.2)

    # Use 127.0.0.1 instead of localhost for better IFrame compatibility
    base_url = f"http://127.0.0.1:{port}"

    # Verify server is responding
    try:
        with urllib.request.urlopen(f"{base_url}/", timeout=2):
            pass  # Server is running
    except Exception as e:
        print(f"Warning: Server may not be ready: {e}")

    # Create URL with playback path pointing to the served recording
    player_url = f"{base_url}/?playbackPath=/recording.viser"
    print(f"Player URL: {player_url}")

    return display(IFrame(src=player_url, width=width, height=height))

