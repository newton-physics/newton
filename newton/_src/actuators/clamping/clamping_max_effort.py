# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar

import warp as wp

from ...utils.deprecation import warn_deprecation
from ..space import DriveOutputSpace
from .clamping_symmetric import ClampingSymmetric


class ClampingMaxEffort(ClampingSymmetric):
    """Deprecated effort-only wrapper for ClampingSymmetric.

    .. deprecated:: 1.9
        Use ``ClampingSymmetric(limit=..., input_space=DriveOutputSpace.EFFORT)``.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if DriveOutputSpace(args.get("input_space", DriveOutputSpace.EFFORT)) is not DriveOutputSpace.EFFORT:
            raise ValueError("ClampingMaxEffort accepts only EFFORT inputs; use ClampingSymmetric for duty")
        resolved = ClampingSymmetric.resolve_arguments({"limit": args.get("max_effort", float("inf"))})
        return {"max_effort": resolved["limit"]}

    def __init__(self, max_effort: wp.array[float]):
        """Initialize the legacy effort-only clamp with limits [N or N·m]."""
        warn_deprecation("ClampingMaxEffort is deprecated in Newton 1.9; use ClampingSymmetric instead.")
        super().__init__(limit=max_effort)

    @property
    def max_effort(self) -> wp.array[float]:
        """Legacy name for the effort limit array [N or N·m]."""
        return self.limit

    @max_effort.setter
    def max_effort(self, value: wp.array[float]) -> None:
        self.limit = value
