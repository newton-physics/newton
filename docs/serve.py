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

"""
Simple HTTP server with proper MIME types for WASM and JS module files.

This script is used to test the built Sphinx documentation locally,
including the Viser static viewer which requires correct MIME types.

Usage:
    # From the repository root:
    python docs/serve.py

    # Or specify a custom port:
    python docs/serve.py --port 8080

Then open http://localhost:8000 in your browser.
"""

import argparse
import http.server
import mimetypes
import os
import sys
from pathlib import Path
from typing import ClassVar

# Add/override MIME types for proper module loading
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with proper MIME type support and CORS headers."""

    # Explicit extensions map for strict MIME type checking
    extensions_map: ClassVar[dict[str, str]] = {  # pyright: ignore[reportIncompatibleVariableOverride]
        ".wasm": "application/wasm",
        ".js": "application/javascript",
        ".css": "text/css",
        ".html": "text/html",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".hdr": "image/vnd.radiance",  # HDR image format for viser HDRI backgrounds
        ".txt": "text/plain",
        ".viser": "application/octet-stream",
        ".ttf": "font/ttf",
        "": "application/octet-stream",
    }

    def guess_type(self, path):
        """Guess the MIME type of a file with proper WASM/JS support."""
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        mimetype, _ = mimetypes.guess_type(path)
        if mimetype is None:
            return "application/octet-stream"
        return mimetype

    def end_headers(self):
        """Add CORS headers for local development."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Serve the built Sphinx documentation with proper MIME types.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=None,
        help="Directory to serve (default: docs/_build/html)",
    )
    args = parser.parse_args()

    # Determine the directory to serve
    if args.directory:
        serve_dir = Path(args.directory)
    else:
        # Default to docs/_build/html relative to this script
        script_dir = Path(__file__).parent
        serve_dir = script_dir / "_build" / "html"

    if not serve_dir.exists():
        print(f"Error: Directory {serve_dir} does not exist.")
        print("Please build the documentation first with:")
        print("  uv run --extra docs --extra sim sphinx-build -b html docs docs/_build/html")
        sys.exit(1)

    os.chdir(serve_dir)

    with http.server.HTTPServer(("", args.port), CORSHTTPRequestHandler) as httpd:
        print(f"Serving documentation at: http://localhost:{args.port}")
        print(f"Directory: {serve_dir.absolute()}")
        print()
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
