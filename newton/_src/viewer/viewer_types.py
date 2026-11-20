# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal typing contracts shared by workspace adapters and inspectors."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeAlias

# Resource pages and candidate pairs use compound identities, not array indices.
TreeKey: TypeAlias = tuple[str, Hashable]
