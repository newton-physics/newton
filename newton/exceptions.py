# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warning categories emitted by Newton.

Newton emits user-actionable warnings with :mod:`warnings` using the categories
below, so applications can filter them by category instead of by message.
"""

from ._src.exceptions import NewtonDeprecationWarning, NewtonWarning

__all__ = [
    "NewtonDeprecationWarning",
    "NewtonWarning",
]
