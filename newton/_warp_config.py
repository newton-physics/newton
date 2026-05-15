# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for Warp logging configuration.

Important:
    Remove this forwarding layer once Newton requires ``warp-lang>=1.14.0``.
"""

from typing import Any

import warp as wp

_DEPRECATED_LOG_CONFIG_KEYS = {"quiet", "verbose"}


def _has_log_level_api(warp_module: Any = wp) -> bool:
    return (
        hasattr(warp_module.config, "log_level")
        and hasattr(warp_module, "LOG_DEBUG")
        and hasattr(warp_module, "LOG_INFO")
        and hasattr(warp_module, "LOG_WARNING")
    )


def warp_config_hasattr(name: str, warp_module: Any = wp) -> bool:
    """Return whether ``warp.config`` supports *name* without touching deprecated log keys."""
    if name in _DEPRECATED_LOG_CONFIG_KEYS and _has_log_level_api(warp_module):
        return True
    return hasattr(warp_module.config, name)


def warp_verbose_enabled(warp_module: Any = wp) -> bool:
    """Return whether Warp-style verbose/debug logging is enabled."""
    legacy_verbose = bool(vars(warp_module.config).get("verbose", False))
    if _has_log_level_api(warp_module):
        return legacy_verbose or warp_module.config.log_level <= warp_module.LOG_DEBUG
    return legacy_verbose


def set_warp_quiet(enabled: bool = True, warp_module: Any = wp) -> None:
    """Suppress Warp info-level output, preserving compatibility with older Warp versions."""
    if _has_log_level_api(warp_module):
        if enabled:
            if warp_module.config.log_level < warp_module.LOG_WARNING:
                warp_module.config.log_level = warp_module.LOG_WARNING
        elif warp_module.config.log_level == warp_module.LOG_WARNING:
            warp_module.config.log_level = warp_module.LOG_INFO
        return

    warp_module.config.quiet = enabled


def set_warp_config_value_compat(name: str, value: Any, warp_module: Any = wp) -> None:
    """Set a ``warp.config`` value, translating legacy ``verbose``/``quiet`` keys."""
    if _has_log_level_api(warp_module):
        if name == "verbose":
            if value:
                if warp_module.config.log_level > warp_module.LOG_DEBUG:
                    warp_module.config.log_level = warp_module.LOG_DEBUG
            elif warp_module.config.log_level == warp_module.LOG_DEBUG:
                warp_module.config.log_level = warp_module.LOG_INFO
            return

        if name == "quiet":
            set_warp_quiet(bool(value), warp_module=warp_module)
            return

    setattr(warp_module.config, name, value)
