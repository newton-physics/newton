# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal typing contracts shared by workspace adapters and inspectors."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from .viewer import ViewerBase

# Resource pages and candidate pairs use compound identities, not array indices.
TreeKey: TypeAlias = tuple[str, Hashable]


class ViewerExample(Protocol):
    """Minimum example contract; optional simulation data is discovered separately."""

    viewer: ViewerBase

    def step(self) -> None:
        """Advance the live simulation."""
        ...

    def render(self) -> None:
        """Log the current display frame."""
        ...
