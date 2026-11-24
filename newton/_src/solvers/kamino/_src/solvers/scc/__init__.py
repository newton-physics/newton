# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split Constraint Coupling (SCC) solver for Kamino's dual forward-dynamics problem."""

from .solver import SCCSolver
from .types import SCCConfigStruct, SCCData, SCCInfo, SCCState, SCCStatus, convert_config_to_struct

__all__ = [
    "SCCConfigStruct",
    "SCCData",
    "SCCInfo",
    "SCCSolver",
    "SCCState",
    "SCCStatus",
    "convert_config_to_struct",
]
