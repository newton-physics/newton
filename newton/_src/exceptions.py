# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


class NewtonWarning(UserWarning):
    """Category for Newton warnings other than deprecations.

    Filter on this category to control these warnings at once, for example
    ``warnings.filterwarnings("error", category=newton.exceptions.NewtonWarning)``.
    Deprecations use :class:`NewtonDeprecationWarning` instead.
    """


class NewtonDeprecationWarning(DeprecationWarning):
    """Category for deprecated Newton features.

    It does not subclass :class:`NewtonWarning`, so :class:`UserWarning`
    filters do not affect deprecations. Python hides
    :class:`DeprecationWarning` subclasses by default outside ``__main__``;
    enable them with ``-W default::DeprecationWarning`` or
    :func:`warnings.simplefilter`.
    """
