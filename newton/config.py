# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Global cache settings, read when downloading assets or approximating meshes.

Asset caching is enabled by default; mesh decomposition caching is opt-in.
The initial asset cache location comes from ``NEWTON_CACHE_PATH`` or the
platform's user cache directory for Newton. An empty ``NEWTON_CACHE_PATH``
uses the current working directory for assets. Set the environment variable
before importing Newton, or assign these settings before the next download or
mesh approximation (including USD import).

Each setting accepts a string or path-like directory; ``None`` disables that
cache. An explicit ``cache_dir`` directory on a call overrides its global
setting. Passing ``cache_dir=None`` uses the global setting.
"""

import os

from warp._src.thirdparty.appdirs import user_cache_dir

__all__ = ["cache_dir_assets", "cache_dir_meshes"]

_cache_dir = os.environ.get("NEWTON_CACHE_PATH")
if _cache_dir is None:
    _cache_dir = user_cache_dir("newton", "newton-physics")

cache_dir_assets: str | os.PathLike[str] | None = _cache_dir
"""Directory for downloaded assets, including ``newton-assets``.

Defaults to the Newton cache directory, preserving existing downloads. Set to
``None`` to download each request into a fresh temporary directory instead of
reusing persistent files. Temporary assets remain available until process exit.
"""

cache_dir_meshes: str | os.PathLike[str] | None = None
"""Directory for CoACD and V-HACD convex mesh decompositions.

Defaults to ``None`` (no disk caching). Set a directory to enable caching,
including automatic USD mesh approximation; set ``None`` to disable it again.
This setting does not affect :meth:`newton.Mesh.build_sdf` caching.
"""
