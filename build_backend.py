# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Delegate to Hatchling without wheel-only tool requirements for editables.

Hatchling otherwise returns the wheel hook's rendering dependencies for PEP 660
too, before the hook can skip rendering. Editable builds only need Hatchling's
lightweight editable helper.
"""

from hatchling import build as _backend


def __getattr__(name):
    # Hatchling conditionally omits metadata hooks under pip. Preserve that
    # availability, so pip can use its ordinary build-wheel fallback.
    return getattr(_backend, name)


def get_requires_for_build_editable(config_settings=None):
    """Return Hatchling's editable helper without documentation dependencies."""
    # Hatchling uses this small helper to construct its PEP 660 wheel.
    return ["editables~=0.3"]
