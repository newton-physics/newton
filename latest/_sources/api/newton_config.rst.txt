.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.config
=============

Global cache settings, read when downloading assets or approximating meshes.

Asset caching is enabled by default; mesh decomposition caching is opt-in.
The initial asset cache location comes from ``NEWTON_CACHE_PATH`` or the
platform's user cache directory for Newton. An empty ``NEWTON_CACHE_PATH``
uses the current working directory for assets. Set the environment variable
before importing Newton, or assign these settings before the next download or
mesh approximation (including USD import).

Each setting accepts a string or path-like directory; ``None`` disables that
cache. An explicit ``cache_dir`` directory on a call overrides its global
setting. Passing ``cache_dir=None`` uses the global setting.

.. py:module:: newton.config
.. currentmodule:: newton.config

.. rubric:: Attributes

.. autodata:: cache_dir_assets
   :no-value:

.. autodata:: cache_dir_meshes
   :no-value:
