# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warnings and exceptions raised by Newton's public API."""


class NewtonWarning(UserWarning):
    """Base class for user-actionable Newton warnings."""


class NewtonDeprecationWarning(NewtonWarning, DeprecationWarning):
    """Warning category for deprecated Newton APIs."""


class NewtonGeometryWarning(NewtonWarning):
    """Warning category for recoverable geometry issues."""


__all__ = [
    "NewtonDeprecationWarning",
    "NewtonGeometryWarning",
    "NewtonWarning",
]
