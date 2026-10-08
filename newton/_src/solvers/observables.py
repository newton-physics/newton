# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standard solver observable kinds."""

from enum import Enum


class SolverObservableKind(Enum):
    """Standard observable quantities that may be requested from a solver.

    Requests are composed as a :class:`set` rather than a bit mask so that
    solver-specific enums can add entries without coordinating integer bits
    with Newton or other solver implementations.

    Containers associate kinds with arrays using :meth:`SolverBase.Observables.field`.
    Kind values are identifiers, not array names or allocation instructions.

    .. experimental::

        The solver observable API may change while additional solvers and observable
        categories are migrated to it.
    """

    BODY_QDD = "body_qdd"
    """Rigid-body spatial accelerations."""

    BODY_PARENT_F = "body_parent_f"
    """Incoming parent-joint wrenches on rigid bodies."""

    CONTACT_F = "contact_f"
    """Spatial contact forces aligned with a :class:`~newton.Contacts` container."""
