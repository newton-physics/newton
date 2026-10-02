# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


class NewtonWarning(UserWarning):
    """Base category for warnings emitted by Newton.

    Filter on this category to control all Newton warnings at once, for
    example ``warnings.filterwarnings("error", category=newton.exceptions.NewtonWarning)``.
    """


class NewtonDeprecationWarning(NewtonWarning, DeprecationWarning):
    """Category for deprecated Newton features.

    Python hides :class:`DeprecationWarning` subclasses by default outside
    ``__main__``; enable them with ``-W default::DeprecationWarning`` or
    :func:`warnings.simplefilter`.
    """
