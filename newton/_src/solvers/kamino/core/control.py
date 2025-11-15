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

"""Defines the Control container for Kamino."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@dataclass
class Control:
    """
    Time-varying control data for a :class:`Model`.

    Time-varying control data currently consists of generalized joint actuation forces, with
    the intention that external actuator models or controllers will populate these attributes.

    The exact attributes depend on the contents of the model. Control objects
    should generally be created using the :func:`kamino.Model.control()` function.

    We adopt the following notational conventions for the control attributes:
    - Generalized joint actuation forces are denoted by ``tau``
    - Subscripts ``_j`` denote joint-indexed quantities, e.g. :attr:`tau_j`.
    """

    tau_j: wp.array | None = None
    """
    Array of generalized joint actuation forces.\n
    Shape is ``(sum(d_j),)`` and dtype is :class:`float32`,\n
    where ``d_j`` is the number of DoFs of each joint ``j``.
    """

    def copy_to(self, other: Control) -> None:
        """
        Copies the Control data to another Control object.

        Args:
            other: The target Control object to copy data into.
        """
        wp.copy(other.tau_j, self.tau_j)

    def copy_from(self, other: Control) -> None:
        """
        Copies the Control data from another Control object.

        Args:
            other: The source Control object to copy data from.
        """
        wp.copy(self.tau_j, other.tau_j)
